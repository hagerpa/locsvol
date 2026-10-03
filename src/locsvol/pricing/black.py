"""Black's formula and the implied volatility (Black, J. Financ. Econ. 3, 1976).

Prices are divided by the discount factor and the forward F. For a call (s = 1) or a put
(s = -1) with log strike k = log(K/F) and total variance w = sigma^2 T,

    p(w, k) = s (Phi(s d_1) - e^k Phi(s d_2)),    d_1 = -k / sqrt(w) + sqrt(w) / 2,
                                                   d_2 = d_1 - sqrt(w),

and the vega, the derivative of p in sigma, is sqrt(T) phi(d_1) for calls and puts alike.

The implied volatility solves p(sigma^2 T, k) = P for sigma in [0, sigma_max]. Since p increases
in sigma from the intrinsic value (s (1 - e^k))^+ at sigma = 0, the bracket is halved until both
the price residual and the half-width of the bracket are below their tolerances. A price below
the intrinsic value, above p(sigma_max^2 T, k), or not reached within the iterations has no
implied volatility in the bracket and gives nan.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr


@jax.jit
def black_price(total_variance, log_strike, is_call):
    """p(w, k) for w > 0, calls where is_call and puts elsewhere."""
    root = jnp.sqrt(total_variance)
    d1 = -log_strike / root + 0.5 * root
    d2 = d1 - root
    sign = jnp.where(is_call, 1.0, -1.0)
    return sign * (ndtr(sign * d1) - jnp.exp(log_strike) * ndtr(sign * d2))


@jax.jit
def black_vega(volatility, maturity, log_strike):
    """sqrt(T) phi(d_1), the derivative of the price in the volatility."""
    root = jnp.sqrt(maturity)
    total = volatility * root
    d1 = -log_strike / total + 0.5 * total
    return root * jnp.exp(-0.5 * d1**2) / jnp.sqrt(2.0 * jnp.pi)


@partial(jax.jit, static_argnames="iterations")
def implied_volatility(
    price,
    maturity,
    log_strike,
    is_call,
    maximum_volatility=5.0,
    price_tolerance=1e-12,
    volatility_tolerance=1e-12,
    iterations=100,
):
    """sigma in [0, maximum_volatility] with p(sigma^2 T, k) = price, by bisection; nan where
    no such sigma is found."""
    price, maturity, log_strike, is_call = jnp.broadcast_arrays(
        *(jnp.asarray(v, dtype=jnp.float64) for v in (price, maturity, log_strike)),
        jnp.asarray(is_call),
    )
    strike = jnp.exp(log_strike)
    valid = jnp.isfinite(price) & jnp.isfinite(maturity) & (maturity > 0) & jnp.isfinite(strike)
    maturity = jnp.where(valid, maturity, 1.0)
    log_strike = jnp.where(valid, log_strike, 0.0)
    intrinsic = jnp.where(is_call, jnp.maximum(1.0 - strike, 0.0), jnp.maximum(strike - 1.0, 0.0))
    supremum = jnp.where(is_call, 1.0, strike)
    bracket = black_price(maximum_volatility**2 * maturity, log_strike, is_call)
    inside = valid & (price >= intrinsic) & (price <= bracket) & (price < supremum)
    at_zero = inside & (price == intrinsic)
    at_maximum = inside & ~at_zero & (price == bracket)
    search = inside & ~at_zero & ~at_maximum

    def model_price(sigma):
        variance = sigma**2 * maturity
        positive = jnp.isfinite(variance) & (variance > 0)
        value = black_price(jnp.where(positive, variance, 1.0), log_strike, is_call)
        return jnp.where(positive, value, intrinsic)

    def halve(_, bounds):
        lower, upper = bounds
        middle = 0.5 * (lower + upper)
        below = search & (model_price(middle) < price)
        return jnp.where(below, middle, lower), jnp.where(search & ~below, middle, upper)

    start = (jnp.zeros_like(price), jnp.full_like(price, maximum_volatility))
    lower, upper = jax.lax.fori_loop(0, iterations, halve, start)
    sigma = 0.5 * (lower + upper)
    converged = (jnp.abs(model_price(sigma) - price) <= price_tolerance) & (
        0.5 * (upper - lower) <= volatility_tolerance
    )
    sigma = jnp.where(at_zero, 0.0, jnp.where(at_maximum, maximum_volatility, sigma))
    return jnp.where(at_zero | at_maximum | (search & converged), sigma, jnp.nan)
