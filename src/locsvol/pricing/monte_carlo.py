"""Monte Carlo prices of European options and their pathwise derivatives.

Prices are divided by the discount factor and the forward of their maturity. For a call (s = 1)
or a put (s = -1) with maturity T and log strike k = log(K/F), and paths X^1, ..., X^m of
X = log(S/F) from the Euler scheme on a grid that contains T,

    P = E[(s (e^{X_T} - e^k))^+]  ~  (1/m) sum_n (s (e^{X^n_T} - e^k))^+.

The paths of one scrambled Sobol sequence are dependent, so independent scramblings are averaged
as replications.

Derivatives are pathwise (Broadie and Glasserman, Manag. Sci. 42, 1996): for a parameter beta
that moves the paths,

    dP/dbeta  ~  (1/m) sum_n s e^{X^n_T} 1{s (e^{X^n_T} - e^k) > 0} dX^n_T/dbeta.

For the coefficients beta of the leverage factor, the derivatives D_j = dX_j/dbeta follow by
differentiating the Euler scheme X_{j+1} = X_j - sigma_j^2 h_j / 2 + sigma_j dW_j with
sigma_j = L(t_j, X_j) sqrt(Y_j) and h_j = t_{j+1} - t_j along each path,

    D_{j+1} = D_j + (dW_j - sigma_j h_j) sqrt(Y_j) (d_x L(t_j, X_j) D_j + d_beta L(t_j, X_j)),
    D_0 = 0,

where d_beta L = L b for a parametrized leverage factor L = L_0 exp(b^T beta). The derivative
d_x L is taken once per path and step, so each further coefficient adds only a few arithmetic
operations per path and step.

The sums over paths run over chunks of paths, so that memory stays bounded.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

CHUNK = 1024  # paths per chunk


def maturity_nodes(times, maturities):
    """For each maturity the index j with times[j] equal to it."""
    t, m = np.asarray(times), np.asarray(maturities)
    nodes = np.minimum(np.searchsorted(t, m), t.size - 1)
    if not np.array_equal(t[nodes], m):
        raise ValueError("every maturity must be a node of the time grid")
    return nodes


def maturity_groups(nodes):
    """The distinct nodes and, for each, the indices of the quotes that mature there."""
    unique = np.unique(nodes)
    return unique, [np.flatnonzero(nodes == node) for node in unique]


@jax.jit
def payoffs(x, log_strikes, is_call):
    """(s (e^x - e^k))^+ with s = 1 for calls and s = -1 for puts, broadcast over x and k."""
    sign = jnp.where(is_call, 1.0, -1.0)
    return jnp.maximum(sign * (jnp.exp(x) - jnp.exp(log_strikes)), 0.0)


def monte_carlo_prices(paths, leverage, maturities, log_strikes, is_call):
    """Prices (quotes,) of the model with the leverage factor, from the paths of one stream."""
    nodes = maturity_nodes(paths.times, maturities)
    x = paths.log_spot(leverage)
    k, call = jnp.asarray(log_strikes), jnp.asarray(is_call)
    prices = jnp.zeros(k.size)
    for node, rows in zip(*maturity_groups(nodes), strict=True):
        prices = prices.at[rows].set(jnp.mean(payoffs(x[:, node, None], k[rows], call[rows]), 0))
    return prices


def replicated_prices(paths, leverage, maturities, log_strikes, is_call):
    """The mean of the Monte Carlo prices over independent replications, a sequence of paths."""
    return jnp.mean(
        jnp.stack(
            [monte_carlo_prices(p, leverage, maturities, log_strikes, is_call) for p in paths]
        ),
        axis=0,
    )


@jax.jit
def log_spot_derivatives(leverage, times, variance, spot_increments, nodes):
    """X at all nodes (paths, nodes) and D = dX/dbeta at the given nodes (nodes, paths, size).

    beta are the coefficients of the leverage factor; see the module docstring. A path becomes
    nan from the first step with an invalid value.
    """
    coefficients = leverage.coefficients
    slots = jnp.full(times.size, -1).at[nodes].set(jnp.arange(nodes.size))

    def coefficient_derivative(t, x):
        return jax.jacfwd(lambda c: replace(leverage, coefficients=c)(t, x))(coefficients)

    def step(carry, j):
        x, d, saved = carry
        dt, dw = times[j + 1] - times[j], spot_increments[:, j]
        y = variance[:, j]
        valid = jnp.isfinite(y) & (y >= 0)
        root = jnp.sqrt(jnp.where(valid, y, 0.0))
        value, slope = jax.jvp(lambda z: leverage(times[j], z), (x,), (jnp.ones_like(x),))
        sigma = value * root
        dsigma = (slope[:, None] * d + coefficient_derivative(times[j], x)) * root[:, None]
        following = x - 0.5 * sigma**2 * dt + sigma * dw
        derivative = d - sigma[:, None] * dsigma * dt + dsigma * dw[:, None]
        valid &= jnp.isfinite(x) & jnp.isfinite(sigma) & (sigma >= 0) & jnp.isfinite(following)
        following = jnp.where(valid, following, jnp.nan)
        derivative = jnp.where(valid[:, None], derivative, jnp.nan)
        saved = jax.lax.cond(
            slots[j + 1] >= 0, lambda s: s.at[slots[j + 1]].set(derivative), lambda s: s, saved
        )
        return (following, derivative, saved), following

    x0 = jnp.zeros(variance.shape[0])
    d0 = jnp.zeros((x0.size, coefficients.size))
    saved = jnp.zeros((nodes.size, x0.size, coefficients.size))
    (_, _, saved), values = jax.lax.scan(
        step, (x0, d0, saved), jnp.arange(times.size - 1, dtype=jnp.int32)
    )
    return jnp.concatenate((x0[:, None], values.T), axis=1), saved


@jax.jit
def mean_payoffs_and_derivatives(x, d, log_strikes, is_call):
    """Mean payoffs (quotes,) and their pathwise derivatives (quotes, size) for terminal
    values x (paths,) with derivatives d (paths, size), for quotes of one maturity."""
    value, slope = jax.jvp(
        lambda z: payoffs(z[:, None], log_strikes, is_call), (x,), (jnp.ones_like(x),)
    )
    return jnp.mean(value, axis=0), slope.T @ d / x.size


def chunks(count, size=CHUNK):
    """Consecutive slices of at most size paths covering count paths."""
    return [slice(start, min(start + size, count)) for start in range(0, count, size)]
