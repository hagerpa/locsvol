"""The out-of-the-money Black price and its inversion in total variance.

The out-of-the-money option is the put for k = log(K/F) < 0 and the call for k >= 0. Its Black
price BS(k, w), divided by the discount factor and the forward (pricing.black), increases
strictly in the total variance w, from zero at w = 0. A price p between the prices at
w_lo <= w_hi is attained at a unique total variance in [w_lo, w_hi], found by bisection until the
bracket is a single floating point interval. Its derivatives follow from the implicit function
theorem,

    dw = (dp - BS_k dk) / BS_w,

so the inverted total variance can be differentiated twice, as Dupire's formula requires.
"""

import jax
import jax.numpy as jnp

from locsvol.pricing.black import black_price

ITERATIONS = 80  # bisection steps
PRICE_TOLERANCE = 1e-12  # |BS(k, w) - p| at the returned w
VARIANCE_TOLERANCE = 1e-12  # half the final bracket


def otm_price(total_variance, log_strike):
    """BS(k, w) of the out-of-the-money option, zero at w = 0."""
    positive = total_variance > 0.0
    price = black_price(jnp.where(positive, total_variance, 1.0), log_strike, log_strike >= 0.0)
    return jnp.where(positive, price, 0.0)


@jax.custom_jvp
def otm_total_variance(price, log_strike, lower, upper, lower_price, upper_price):
    """The w in [lower, upper] with otm_price(w, k) = price, given the prices at both ends.

    nan where the bracket is invalid, the price lies outside the bracket's prices, or the
    bisection misses its tolerances.
    """
    valid = (
        jnp.isfinite(price)
        & jnp.isfinite(log_strike)
        & jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & (lower >= 0.0)
        & (upper > 0.0)
        & (lower <= upper)
        & jnp.isfinite(lower_price)
        & jnp.isfinite(upper_price)
        & (lower_price >= 0.0)
        & (lower_price <= upper_price)
    )
    k = jnp.where(valid, log_strike, 0.0)
    left, right = jnp.where(valid, lower, 0.0), jnp.where(valid, upper, 1.0)
    valid &= (price >= lower_price) & (price <= upper_price)

    def unresolved(left, right):
        middle = left + 0.5 * (right - left)
        return (middle != left) & (middle != right)

    def step(carry):
        index, left, right = carry
        middle = left + 0.5 * (right - left)
        moving = unresolved(left, right)
        below = otm_price(middle, k) < price
        left = jnp.where(moving & below, middle, left)
        right = jnp.where(moving & ~below, middle, right)
        return index + 1, left, right

    def running(carry):
        index, left, right = carry
        return (index < ITERATIONS) & jnp.any(unresolved(left, right))

    _, left, right = jax.lax.while_loop(running, step, (0, left, right))
    candidate = left + 0.5 * (right - left)
    converged = (jnp.abs(otm_price(candidate, k) - price) <= PRICE_TOLERANCE) & (
        0.5 * (right - left) <= VARIANCE_TOLERANCE
    )
    at_lower, at_upper = price == lower_price, price == upper_price
    value = jnp.where(at_lower, lower, jnp.where(at_upper, upper, candidate))
    return jnp.where(valid & (at_lower | at_upper | converged), value, jnp.nan)


@otm_total_variance.defjvp
def _otm_total_variance_jvp(primals, tangents):
    """dw = (dp - BS_k dk) / BS_w; the brackets are search aids and carry no derivative."""
    price, k = primals[:2]
    price_tangent, k_tangent = tangents[:2]
    w = otm_total_variance(*primals)
    valid = jnp.isfinite(w) & (w > 0.0)
    safe_w, safe_k = jnp.where(valid, w, 1.0), jnp.where(valid, k, 0.0)
    call = safe_k >= 0.0
    _, price_w = jax.jvp(lambda v: black_price(v, safe_k, call), (safe_w,), (jnp.ones_like(w),))
    _, price_k = jax.jvp(lambda x: black_price(safe_w, x, call), (safe_k,), (jnp.ones_like(k),))
    valid &= jnp.isfinite(price_w) & (price_w > 0.0) & jnp.isfinite(price_k)
    tangent = (price_tangent - price_k * k_tangent) / jnp.where(valid, price_w, 1.0)
    return w, jnp.where(valid, tangent, jnp.nan)
