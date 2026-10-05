import jax.numpy as jnp
import numpy as np

import locsvol  # noqa: F401
from locsvol.calibration.correction import (
    correction_steps,
    path_gram,
    sensitivities,
    support_points,
    widened_intervals,
)
from locsvol.calibration.fixed_point import trapezoid_weights
from locsvol.calibration.gauss_newton import LocalSolver, interval_residuals
from locsvol.leverage.chart import Chart
from locsvol.leverage.parametrized import Parametrized
from locsvol.pricing.monte_carlo import replicated_prices
from locsvol.regression import ChebyshevTensor
from locsvol.simulation.paths import Paths
from locsvol.simulation.random import brownian_increments

TIMES = jnp.linspace(0.0, 1.0, 17)
CHART = Chart(jnp.array([0.0, 2.0]), jnp.array([0.0, 0.08]), 1.0, 0.5, (0.0, 1.0)).with_envelope(
    jnp.array([0.0, 1.0]), jnp.array([-0.3, -0.6]), jnp.array([0.3, 0.6]), float(TIMES[1])
)


def flat(t, x):
    """The constant volatility 0.2."""
    return jnp.full(jnp.broadcast_shapes(jnp.shape(t), jnp.shape(x)), 0.2)


def replications(count=256):
    paths = []
    for r in range(2):
        increments = brownian_increments(TIMES, {"seed": 11, "replication": r}, count, 2)
        variance = jnp.exp(0.3 * jnp.cumsum(jnp.pad(increments[:, :, 1], ((0, 0), (1, 0))), axis=1))
        paths.append(Paths(TIMES, variance / jnp.mean(variance, axis=0), increments[:, :, 0]))
    return paths


def leverage(coefficients=None):
    basis = ChebyshevTensor((1, 2))
    coefficients = jnp.zeros(basis.size) if coefficients is None else coefficients
    return Parametrized(flat, CHART, basis, coefficients)


def test_widened_intervals():
    bid, ask = jnp.array([1.0, 1.0, 1.0]), jnp.array([2.0, 2.0, 2.0])
    lower, upper = widened_intervals(
        bid, ask, jnp.array([True, False, False]), jnp.array([0.5, 0.5, 2.5])
    )
    np.testing.assert_array_equal(lower, [1.0, 0.5, 1.0])
    np.testing.assert_array_equal(upper, [2.0, 2.0, 2.5])


def test_support_points_span_the_envelope():
    t, x = support_points(CHART, TIMES, 9, 5)
    points = np.stack((t, x), axis=1)
    np.testing.assert_array_equal(points, np.unique(points, axis=0))
    assert np.unique(np.asarray(t)).size == 9
    v = np.asarray(CHART(t, x)[:, 1])
    assert np.all(np.abs(v) <= 1 + 1e-12)
    for time in np.unique(np.asarray(t))[1:]:
        np.testing.assert_allclose(
            np.sort(v[np.asarray(t) == time]), np.linspace(-1, 1, 5), atol=1e-12
        )
    np.testing.assert_allclose(CHART(1.0, CHART.log_moneyness(1.0, 0.4))[1], 0.4 / 0.6, atol=1e-14)


def test_path_gram_equals_the_dense_sum():
    lev = leverage(0.1 * jnp.cos(jnp.arange(6.0)))
    x = replications()[0].log_spot(lev)
    weights = trapezoid_weights(TIMES)
    design = lev.basis(lev.chart(TIMES[None, :], x))
    dense = jnp.einsum("pna,pnb,n->ab", design, design, weights) / x.shape[0]
    np.testing.assert_allclose(path_gram(lev, TIMES, x, weights), dense, atol=1e-14)


def market():
    paths = replications()
    maturities = jnp.repeat(jnp.array([0.5, 1.0]), 4)
    k = jnp.tile(jnp.array([-0.1, 0.0, 0.05, 0.1]), 2)
    call = k >= 0
    truth = leverage(jnp.array([0.1, 0.05, -0.08, 0.02, 0.0, 0.03]))
    quoted = replicated_prices(paths, truth, maturities, k, call)
    return paths, maturities, k, call, quoted - 2e-4, quoted + 2e-4


def test_correction_steps_decrease_the_loss_within_the_constraints():
    paths, maturities, k, call, bid, ask = market()
    start, weights = leverage(), trapezoid_weights(TIMES)
    quote_weights, vegas = jnp.full(k.size, 1 / k.size), jnp.full(k.size, 0.2)
    lower, upper = bid, ask
    support = support_points(CHART, TIMES, 9, 5)
    corrected = correction_steps(
        start,
        paths,
        maturities,
        k,
        call,
        lower,
        upper,
        quote_weights,
        vegas,
        weights,
        (0.02, 0.02),
        0.03,
        0.1,
        support,
    )

    def loss(lev):
        prices, _, gram = sensitivities(lev, paths, maturities, k, call, weights)
        r = interval_residuals(prices, lower, upper, quote_weights, vegas)
        return float(r @ r), gram

    losses = [loss(lev)[0] for lev in (start, *corrected)]
    assert losses[1] < losses[0] and losses[2] < losses[1]
    _, initial_gram = loss(start)
    for lev in corrected:
        theta = lev.coefficients - start.coefficients
        assert float(theta @ initial_gram @ theta) <= 0.03**2 * (1 + 1e-7)
        movement = jnp.log(lev(*support)) - jnp.log(start(*support))
        assert float(jnp.max(jnp.abs(movement))) <= 0.1 * (1 + 1e-7)


def test_newton_correction_without_constraints():
    paths, maturities, k, call, bid, ask = market()
    start, weights = leverage(), trapezoid_weights(TIMES)
    quote_weights, vegas = jnp.full(k.size, 1 / k.size), jnp.full(k.size, 0.2)
    solver = LocalSolver(method="newton", tolerance=1e-10, iterations=100)
    (corrected,) = correction_steps(
        start,
        paths,
        maturities,
        k,
        call,
        bid,
        ask,
        quote_weights,
        vegas,
        weights,
        (None,),
        solver=solver,
    )
    before, _, _ = sensitivities(start, paths, maturities, k, call, weights)
    after, _, _ = sensitivities(corrected, paths, maturities, k, call, weights)
    r0 = interval_residuals(before, bid, ask, quote_weights, vegas)
    r1 = interval_residuals(after, bid, ask, quote_weights, vegas)
    assert float(r1 @ r1) < float(r0 @ r0)
