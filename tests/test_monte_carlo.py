from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import locsvol  # noqa: F401
from locsvol.leverage.chart import Chart
from locsvol.leverage.parametrized import Parametrized
from locsvol.pricing.black import black_price
from locsvol.pricing.monte_carlo import (
    log_spot_derivatives,
    maturity_nodes,
    mean_payoffs_and_derivatives,
    monte_carlo_prices,
    payoffs,
    replicated_prices,
)
from locsvol.regression import ChebyshevTensor
from locsvol.simulation.euler import log_spot
from locsvol.simulation.paths import Paths
from locsvol.simulation.random import brownian_increments


def flat(t, x):
    """The constant volatility 0.2."""
    return jnp.full(jnp.broadcast_shapes(jnp.shape(t), jnp.shape(x)), 0.2)


TIMES = jnp.linspace(0.0, 1.0, 17)
CHART = Chart(jnp.array([0.0, 2.0]), jnp.array([0.0, 0.08]), 1.0, 0.5, (0.0, 1.0))


def flat_paths(count, replication=0):
    increments = brownian_increments(TIMES, {"seed": 7, "replication": replication}, count, 1)
    return Paths(TIMES, jnp.ones((count, TIMES.size)), increments[:, :, 0])


def leverage(size=(2, 3)):
    basis = ChebyshevTensor(size)
    coefficients = 0.05 * jnp.sin(jnp.arange(basis.size, dtype=jnp.float64))
    return Parametrized(flat, CHART, basis, coefficients)


def test_maturity_nodes():
    np.testing.assert_array_equal(maturity_nodes(TIMES, [0.5, 1.0, 0.5]), [8, 16, 8])
    with pytest.raises(ValueError):
        maturity_nodes(TIMES, [0.3])


def test_payoffs_of_calls_and_puts():
    k = jnp.log(jnp.array([0.9, 1.1]))
    values = payoffs(jnp.log(jnp.array([[1.0], [1.2]])), k, jnp.array([True, False]))
    np.testing.assert_allclose(values, [[0.1, 0.1], [0.3, 0.0]], atol=1e-15)


def test_flat_volatility_gives_black_prices():
    maturities = jnp.array([0.25, 0.25, 1.0, 1.0])
    k, call = jnp.array([-0.1, 0.1, 0.0, 0.2]), jnp.array([False, True, True, True])
    prices = replicated_prices([flat_paths(4096, r) for r in range(2)], flat, maturities, k, call)
    np.testing.assert_allclose(prices, black_price(0.04 * maturities, k, call), atol=2e-4)
    single = monte_carlo_prices(flat_paths(4096), flat, maturities, k, call)
    assert single.shape == (4,)


def test_out_of_the_money_estimator_converts_by_parity():
    maturities = jnp.array([1.0, 1.0, 1.0, 1.0])
    k, call = jnp.array([-0.3, -0.3, 0.3, 0.3]), jnp.array([True, False, True, False])
    paths = flat_paths(4096)
    direct = np.asarray(monte_carlo_prices(paths, flat, maturities, k, call))
    converted = np.asarray(
        monte_carlo_prices(paths, flat, maturities, k, call, out_of_the_money=True)
    )
    parity = 1.0 - np.exp(np.asarray(k))
    np.testing.assert_allclose(converted[[1, 2]], direct[[1, 2]], atol=1e-15)
    np.testing.assert_allclose(converted[0], direct[1] + parity[0], atol=1e-15)
    np.testing.assert_allclose(converted[3], direct[2] - parity[3], atol=1e-15)
    np.testing.assert_allclose(converted, direct, atol=1.5e-2)


def test_derivatives_match_forward_mode_differentiation():
    paths, lev = flat_paths(64), leverage()
    nodes = jnp.array([4, 16])
    x, d = log_spot_derivatives(lev, TIMES, paths.variance, paths.spot_increments, nodes)
    np.testing.assert_allclose(x, paths.log_spot(lev), atol=1e-14)

    def spot(c):
        return log_spot(replace(lev, coefficients=c), TIMES, paths.variance, paths.spot_increments)

    jacobian = jax.jacfwd(spot)(lev.coefficients)[:, nodes, :]
    np.testing.assert_allclose(jnp.moveaxis(d, 0, 1), jacobian, atol=1e-13)


def test_price_derivatives_match_forward_mode_differentiation():
    paths, lev = flat_paths(64), leverage()
    k, call = jnp.array([-0.05, 0.0, 0.1]), jnp.array([False, True, True])
    x, d = log_spot_derivatives(lev, TIMES, paths.variance, paths.spot_increments, jnp.array([16]))
    prices, jacobian = mean_payoffs_and_derivatives(x[:, 16], d[0], k, call)

    def price(c):
        terminal = log_spot(
            replace(lev, coefficients=c), TIMES, paths.variance, paths.spot_increments
        )
        return jnp.mean(payoffs(terminal[:, 16, None], k, call), axis=0)

    np.testing.assert_allclose(prices, price(lev.coefficients), atol=1e-15)
    np.testing.assert_allclose(jacobian, jax.jacfwd(price)(lev.coefficients), atol=1e-13)
