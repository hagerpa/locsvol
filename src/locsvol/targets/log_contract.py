"""Total variance of the log-contract and forward variance of a total variance surface.

The log-contract pays -2 log(S_T/F); its price, T times the fair variance-swap rate, is the
strip of out-of-the-money options (Demeterfi, Derman, Kamal and Zou, J. Derivatives 6, 1999;
Gatheral, The Volatility Surface, Wiley, 2006, Ch. 11)

    V(T) = 2 int e^{-k} BS(k, w(T, k)) dk,

integrated over the log-moneyness interval [k_lo, k_hi] of the surface, with BS the normalized
Black price of the out-of-the-money option. The integral is computed by Gauss-Legendre quadrature
of the same order on [k_lo, 0] and on [0, k_hi]. On a time grid t_0 < ... < t_m, the forward
variance is the piecewise constant

    xi(t) = (V(t_{j+1}) - V(t_j)) / (t_{j+1} - t_j),    t_j <= t < t_{j+1}.
"""

import jax.numpy as jnp
import numpy as np

from locsvol.targets.inversion import otm_price


def log_contract_quadrature(order=256, lower=-20.0, upper=20.0):
    """Nodes k_j and weights 2 e^{-k_j} v_j of the Gauss-Legendre rule of the given order on
    [lower, 0] and on [0, upper], 2 order of each."""
    x, v = np.polynomial.legendre.leggauss(order)
    nodes = np.concatenate((0.5 * -lower * x + 0.5 * lower, 0.5 * upper * x + 0.5 * upper))
    weights = np.concatenate((0.5 * -lower * v, 0.5 * upper * v))
    return jnp.asarray(nodes), jnp.asarray(2.0 * weights * np.exp(-nodes))


def log_contract_variance(surface, t, order=256, bounds=None):
    """V(t) at the maturities t (any shape), by the quadrature of the given order on bounds, by
    default the surface's log-moneyness bounds."""
    nodes, weights = log_contract_quadrature(order, *(bounds or surface.log_moneyness_bounds))
    t = jnp.asarray(t, dtype=float)
    w = surface.total_variance(t[..., None], nodes)
    return jnp.sum(weights * otm_price(w, nodes), axis=-1)


def forward_variance(surface, times, order=256, bounds=None):
    """xi on the intervals of the grid times, from log_contract_variance."""
    times = jnp.asarray(times, dtype=float)
    return jnp.diff(log_contract_variance(surface, times, order, bounds)) / jnp.diff(times)
