"""The fixed point with time-interpolated coefficients and the progressive calibration."""

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
import pytest

import locsvol  # noqa: F401
from locsvol.backbones.constant import Constant
from locsvol.calibration.progressive import progressive
from locsvol.calibration.time_interpolated import (
    TimeInterpolated,
    full_horizon,
    interval_progressive,
)
from locsvol.leverage.coordinates import Affine
from locsvol.regression.bins import QuantileBins
from locsvol.regression.chebyshev import ChebyshevTensor
from locsvol.regression.gamma import GammaRegression
from locsvol.regression.network import PositiveMeanNetwork
from locsvol.simulation.paths import simulate

TIMES = jnp.linspace(0.0, 0.5, 11)
KNOTS = TIMES[jnp.array([0, 4, 10])]
CHART = Affine((0.0, 0.0), (1.0, 0.2))
BASIS = ChebyshevTensor((0, 2))
REGRESSION = GammaRegression(BASIS, ridge=1e-8)
ROWS = jnp.array([[0.1, -0.2, 0.05], [0.0, 0.1, -0.1], [0.2, 0.0, 0.1]])
X = jnp.linspace(-0.3, 0.3, 7)


def smile(t, x):
    return 0.2 + 0.3 * x**2 + 0.05 * t


@dataclass(frozen=True)
class Independent:
    """Y = 1 +- 0.5 with probability 1/2 each, independent of the spot's Brownian motion."""

    factors = 2

    def variance(self, times, increments):
        sign = jnp.sign(increments[:, :1, 1])
        return jnp.broadcast_to(1.0 + 0.5 * sign, (increments.shape[0], times.size))


def test_coefficients_are_interpolated_linearly_between_knots():
    leverage = TimeInterpolated(smile, CHART, BASIS, KNOTS, ROWS)
    middle = 0.5 * (ROWS[1] + ROWS[2])
    np.testing.assert_allclose(leverage.coefficients_at(0.1), 0.5 * (ROWS[0] + ROWS[1]), atol=1e-16)
    np.testing.assert_allclose(leverage.coefficients_at(0.35), middle, atol=1e-16)
    assert np.all(np.isnan(leverage.coefficients_at(0.6)))
    z = CHART(0.35, X)
    expected = smile(0.35, X) / np.sqrt(np.exp(BASIS(z) @ middle))
    np.testing.assert_allclose(leverage(0.35, X), expected, rtol=1e-14)
    np.testing.assert_allclose(leverage(jnp.full(7, 0.35), X), expected, rtol=1e-14)


@pytest.mark.parametrize("formulation", ["direct", "residual"])
def test_a_constant_variance_factor_leaves_the_local_volatility(formulation):
    paths = simulate(Constant(), TIMES, {"seed": 1}, 256)
    leverage = TimeInterpolated(smile, CHART, BASIS, KNOTS, ROWS)
    final = full_horizon(leverage, paths, REGRESSION, 2, formulation=formulation)[-1]
    np.testing.assert_allclose(final(0.3, X), smile(0.3, X), rtol=1e-7)
    final = interval_progressive(leverage, paths, REGRESSION, 2, 0.5, formulation)
    np.testing.assert_allclose(final(0.0, 0.0), smile(0.0, 0.0), rtol=1e-7)
    np.testing.assert_allclose(final.coefficients[1:], 0.25 * ROWS[1:], atol=1e-7)


def test_relaxation_blends_the_coefficients():
    paths = simulate(Independent(), TIMES, {"seed": 2}, 512)
    leverage = TimeInterpolated(smile, CHART, BASIS, KNOTS, ROWS)
    full = full_horizon(leverage, paths, REGRESSION, 1)[0]
    half = full_horizon(leverage, paths, REGRESSION, 1, relaxation=0.5)[0]
    expected = 0.5 * ROWS + 0.5 * full.coefficients
    np.testing.assert_allclose(half.coefficients, expected, rtol=1e-14, atol=1e-16)


def test_direct_and_residual_updates_agree_without_penalty():
    paths = simulate(Independent(), TIMES, {"seed": 3}, 512)
    leverage = TimeInterpolated(smile, CHART, BASIS, KNOTS, jnp.zeros((3, 3)))
    direct = full_horizon(leverage, paths, REGRESSION, 2)[-1]
    residual = full_horizon(leverage, paths, REGRESSION, 2, formulation="residual")[-1]
    np.testing.assert_allclose(residual(0.25, X), direct(0.25, X), rtol=1e-6)
    direct = interval_progressive(leverage, paths, REGRESSION, 2)
    residual = interval_progressive(leverage, paths, REGRESSION, 2, formulation="residual")
    np.testing.assert_allclose(residual(0.25, X), direct(0.25, X), rtol=1e-6)


def test_invalid_settings_are_refused():
    paths = simulate(Constant(), TIMES, {"seed": 1}, 64)
    leverage = TimeInterpolated(smile, CHART, BASIS, KNOTS, ROWS)
    with pytest.raises(ValueError):
        full_horizon(replace(leverage, knots=jnp.array([0.0, 0.17, 0.5])), paths, REGRESSION, 1)
    with pytest.raises(ValueError):
        full_horizon(leverage, paths, REGRESSION, 1, relaxation=0.0)
    with pytest.raises(ValueError):
        full_horizon(replace(leverage, floor=0.9), paths, REGRESSION, 1, formulation="residual")
    with pytest.raises(ValueError):
        full_horizon(leverage, paths, GammaRegression(ChebyshevTensor((1, 2)), 1e-8), 1)
    with pytest.raises(ValueError):
        progressive(smile, CHART, paths, [REGRESSION] * 3)


def test_progressive_conditional_mean_is_piecewise_constant_in_time():
    leverage = progressive(smile, CHART, simulate(Constant(), TIMES, {"seed": 4}, 256), REGRESSION)
    np.testing.assert_allclose(leverage(0.3, X), smile(0.3, X), rtol=1e-7)
    leverage = progressive(
        smile, CHART, simulate(Independent(), TIMES, {"seed": 4}, 256), REGRESSION
    )
    for t, s in zip(TIMES[:-1], 0.5 * (TIMES[:-1] + TIMES[1:]), strict=True):
        mean = leverage.means(t, CHART(s, X))
        assert np.array_equal(leverage.means(s, CHART(s, X)), mean)
        np.testing.assert_allclose(leverage(s, X), smile(s, X) / np.sqrt(mean), rtol=1e-14)
    assert np.all(np.isnan(leverage(0.6, X)))


def calibration_error(leverage, paths, node):
    """Largest relative error of L(t, x)^2 E[Y_t | X_t = x] / sigma_D(t, x)^2 over central bins."""
    x, y, t = paths.log_spot(leverage)[:, node], paths.variance[:, node], paths.times[node]
    edges = np.quantile(np.asarray(x), np.linspace(0.1, 0.9, 9))
    errors = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (x >= low) & (x < high)
        middle = 0.5 * (low + high)
        value = leverage(t, middle) ** 2 * jnp.mean(y[inside]) / smile(t, middle) ** 2
        errors.append(abs(float(value) - 1.0))
    return max(errors)


def test_progressive_bins_satisfy_the_calibration_condition():
    paths = simulate(Independent(), TIMES, {"seed": 5}, 4096)
    chart = Affine((0.0, 0.0), (1.0, 0.2), with_time=False)
    leverage = progressive(smile, chart, paths, QuantileBins(16, 50))
    assert max(calibration_error(leverage, paths, node) for node in (3, 6, 9)) < 0.08
    network = PositiveMeanNetwork((3,), max_epochs=2, batch_size=128)
    networks = [replace(network, stream={"seed": 10 + j}) for j in range(TIMES.size - 1)]
    leverage = progressive(smile, CHART, simulate(Constant(), TIMES, {"seed": 6}, 256), networks)
    np.testing.assert_allclose(leverage(0.4, X), smile(0.4, X), rtol=1e-3)
