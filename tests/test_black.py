import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

import locsvol  # noqa: F401
from locsvol.pricing.black import black_price, black_vega, implied_volatility


def test_put_call_parity():
    w, k = jnp.array([0.01, 0.04, 0.2]), jnp.array([-0.3, 0.0, 0.25])
    difference = black_price(w, k, True) - black_price(w, k, False)
    np.testing.assert_allclose(difference, 1 - jnp.exp(k), atol=1e-15)


def test_at_the_money_price():
    np.testing.assert_allclose(black_price(0.09, 0.0, True), 2 * ndtr(0.15) - 1, rtol=1e-14)


def test_vega_is_the_derivative_in_the_volatility():
    derivative = jax.grad(lambda sigma: black_price(sigma**2 * 0.7, 0.1, True))(0.25)
    np.testing.assert_allclose(black_vega(0.25, 0.7, 0.1), derivative, rtol=1e-12)


def test_implied_volatility_inverts_calls_and_puts():
    sigma = jnp.array([0.05, 0.2, 0.8, 2.0])
    t, k = jnp.array([0.02, 0.5, 1.0, 3.0]), jnp.array([-0.01, 0.0, 0.3, 1.0])
    for call in (True, False):
        price = black_price(sigma**2 * t, k, call)
        np.testing.assert_allclose(implied_volatility(price, t, k, call), sigma, rtol=1e-10)


def test_implied_volatility_at_and_beyond_the_bracket():
    assert implied_volatility(0.0, 1.0, 0.0, True) == 0.0
    assert implied_volatility(black_price(25.0, 0.0, True), 1.0, 0.0, True) == 5.0
    assert jnp.isnan(implied_volatility(-1e-3, 1.0, 0.0, True))
    assert jnp.isnan(implied_volatility(0.999, 1.0, 0.0, True))
    assert jnp.isnan(implied_volatility(0.1, 0.0, 0.0, True))
