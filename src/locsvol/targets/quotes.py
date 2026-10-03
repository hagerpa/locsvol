"""Forward, discount factor and log-moneyness of option quotes.

Put-call parity C - P = D (F - K) for the prices C and P of a call and a put with strike K
determines the discount factor D and the forward F of an expiry: the weighted least-squares line
C - P = alpha + beta K through the quotes gives D = -beta and F = alpha / D. It is computed from
the strikes and price differences centred at their weighted means, each scaled by its largest
absolute value, so that neither the level of the strikes nor that of the prices costs precision.

A quote's strike enters the targets as its log-moneyness k = log(K/F), and its implied volatility
as that of the out-of-the-money option, the put for k < 0 and the call for k >= 0.
"""

import jax.numpy as jnp


def put_call_parity(strikes, calls, puts, weights):
    """D and F of the weighted least-squares fit of C - P = D (F - K) over one expiry's quotes.

    Quotes with zero weight are ignored; at least two distinct strikes need positive weight.
    """
    strikes, parity = jnp.asarray(strikes, dtype=float), jnp.asarray(calls) - jnp.asarray(puts)
    weights = jnp.asarray(weights, dtype=float)
    active = weights > 0.0
    weights = weights / jnp.max(weights)
    weights = weights / jnp.sum(weights)
    first = jnp.argmax(active)
    strike_offsets, parity_offsets = strikes - strikes[first], parity - parity[first]
    mean_strike = jnp.sum(weights * jnp.where(active, strike_offsets, 0.0))
    mean_parity = jnp.sum(weights * jnp.where(active, parity_offsets, 0.0))
    x = jnp.where(active, strike_offsets - mean_strike, 0.0)
    y = jnp.where(active, parity_offsets - mean_parity, 0.0)
    x_scale, y_scale = jnp.max(jnp.abs(x)), jnp.max(jnp.abs(y))
    x, y = x / x_scale, y / jnp.where(y_scale > 0.0, y_scale, 1.0)
    slope = jnp.sum(weights * x * y) / jnp.sum(weights * x**2) * y_scale / x_scale
    discount = -slope
    forward = (strikes[first] + mean_strike) + (parity[first] + mean_parity) / discount
    return discount, forward


def log_moneyness(strikes, forwards):
    """k = log(K/F)."""
    return jnp.log(jnp.asarray(strikes) / jnp.asarray(forwards))


def otm_implied_volatility(log_moneyness, call_volatilities, put_volatilities):
    """The implied volatility of the out-of-the-money option: the put's for k < 0, else the
    call's."""
    return jnp.where(jnp.asarray(log_moneyness) >= 0.0, call_volatilities, put_volatilities)
