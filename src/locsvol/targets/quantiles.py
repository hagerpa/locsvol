"""Digital probabilities and quantiles of the law of X_t = log(S_t/F) given by a smile.

By Breeden and Litzenberger (J. Business 51, 1978), the distribution function of X_t is minus
the strike derivative of the call price, which for a total variance smile w(k) = w(t, k) is the
price of the digital put

    P(X_t <= k) = Phi(-d2) + phi(d2) w'(k) / (2 sqrt(w)),    d2 = -k / sqrt(w) - sqrt(w) / 2,

with w'(k) by automatic differentiation; where d2 <= 0 it is evaluated as one minus the price
of the digital call. The quantile of probability p in a bracket [k_lo, k_hi] whose probabilities
enclose p is found by bisection, until the probability is within a tolerance of p and the
bracket's half width is below a tolerance; a probability that leaves the probabilities of the
current bracket, which happens only under butterfly arbitrage, gives nan. For a lognormal law with
total variance w the quantiles are exact: exp(-w/2 + sqrt(w) Phi^{-1}(p)) for S_t/F.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri


def _digital(surface, t, k):
    w, w_k = jax.value_and_grad(lambda x: surface.total_variance(t, x))(k)
    root = jnp.sqrt(w)
    d2 = -k / root - 0.5 * root
    correction = jnp.exp(-0.5 * d2**2) * w_k / (2 * jnp.sqrt(2 * jnp.pi) * root)
    # The small tail is assembled before it is subtracted from one.
    probability = jnp.where(d2 <= 0, 1 - (ndtr(d2) - correction), ndtr(-d2) + correction)
    valid = (
        (t > 0.0)
        & jnp.isfinite(w)
        & (w > 0.0)
        & jnp.isfinite(w_k)
        & (probability >= 0.0)
        & (probability <= 1.0)
    )
    return jnp.where(valid, probability, jnp.nan)


@jax.jit
def digital_probability(surface, t, k):
    """P(X_t <= k) at broadcast t > 0 and k, nan where w or w' is invalid."""
    t, k = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(k, dtype=float))
    values = jax.vmap(partial(_digital, surface))(t.ravel(), k.ravel())
    return values.reshape(t.shape)


@partial(jax.jit, static_argnames=("iterations", "tolerance"))
def digital_quantiles(surface, times, probabilities, brackets, iterations=100, tolerance=1e-10):
    """Log-moneyness quantiles (m, q) of the probabilities (q,) at the times (m,), with brackets
    (m, 2); the tolerance applies to the probability and to the half width of the bracket."""

    def quantile(t, bracket, p):
        lower, upper = bracket
        low, high = _digital(surface, t, lower), _digital(surface, t, upper)
        valid = (low <= p) & (p <= high)
        root = jnp.where(p == low, lower, upper)
        done = valid & ((p == low) | (p == high))

        def step(carry):
            count, left, right, low, high, _, bad, _ = carry
            middle = 0.5 * left + 0.5 * right
            value = _digital(surface, t, middle)
            bad = ~jnp.isfinite(value) | (value < low) | (value > high)
            done = ~bad & (jnp.abs(value - p) <= tolerance) & (0.5 * (right - left) <= tolerance)
            moving, below = ~(bad | done), value < p
            return (
                count + 1,
                jnp.where(moving & below, middle, left),
                jnp.where(moving & ~below, middle, right),
                jnp.where(moving & below, value, low),
                jnp.where(moving & ~below, value, high),
                middle,
                bad,
                done,
            )

        def running(carry):
            count, *_, bad, done = carry
            return valid & ~bad & ~done & (count < iterations)

        carry = (0, lower, upper, low, high, root, False, done)
        *_, root, bad, done = jax.lax.while_loop(running, step, carry)
        return jnp.where(valid & ~bad & done, root, jnp.nan)

    def row(t, bracket):
        return jax.vmap(lambda p: quantile(t, bracket, p))(probabilities)

    times, brackets = jnp.asarray(times, dtype=float), jnp.asarray(brackets, dtype=float)
    probabilities = jnp.asarray(probabilities, dtype=float)
    return jax.vmap(row)(times, brackets)


def lognormal_quantiles(total_variance, probabilities):
    """exp(-w/2 + sqrt(w) Phi^{-1}(p)), the quantiles (..., q) of S_t/F for a lognormal law with
    total variance w (...)."""
    w = jnp.asarray(total_variance, dtype=float)[..., None]
    return jnp.exp(-0.5 * w + jnp.sqrt(w) * ndtri(jnp.asarray(probabilities, dtype=float)))
