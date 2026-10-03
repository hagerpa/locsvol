"""Least squares, quantile bins, per-time means, the neural network and custom regressions."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

import locsvol  # noqa: F401
from locsvol.regression.bins import QuantileBins
from locsvol.regression.custom import CustomRegression
from locsvol.regression.least_squares import LeastSquares
from locsvol.regression.means import per_time_means
from locsvol.regression.network import PositiveMeanNetwork
from locsvol.regression.polynomials import Identity

RNG = np.random.default_rng(8)


def test_least_squares_is_weighted_and_truncated():
    design = RNG.normal(size=(300, 4))
    y = design @ np.array([1.0, -2.0, 0.5, 3.0]) + 0.1 * RNG.normal(size=300)
    w = RNG.uniform(0, 2, 300)
    alpha = LeastSquares(Identity(4), 1e-12).fit(design, y, w)
    root = np.sqrt(w)
    expected = np.linalg.lstsq(root[:, None] * design, root * y, rcond=None)[0]
    np.testing.assert_allclose(alpha, expected, rtol=1e-10)
    doubled = np.concatenate([design, design[:, 3:]], axis=1)
    alpha = LeastSquares(Identity(5), 1e-10).fit(doubled, y, w)
    np.testing.assert_allclose(alpha[3], alpha[4], rtol=1e-8)
    np.testing.assert_allclose(alpha[3] + alpha[4], expected[3], rtol=1e-8)


def sorted_bins(x, y, count):
    order = np.argsort(x[:, 0], kind="stable")
    xs, ys = x[order, 0], y[order]
    groups = np.array_split(np.arange(xs.size), count)
    return xs, groups, [xs[g].mean() for g in groups], [ys[g].mean() for g in groups]


def test_quantile_bins_interpolate_the_bin_means():
    x, y = RNG.normal(size=(200, 1)), RNG.gamma(2.0, 1.0, 200)
    q = np.linspace(-3, 3, 50)[:, None]
    xs, groups, centers, means = sorted_bins(x, y, 8)
    mean = QuantileBins(8).fit(x, y, np.ones(200))
    inside = np.clip(q[:, 0], centers[0], centers[-1])
    expected = np.exp(PchipInterpolator(centers, np.log(means))(inside))
    np.testing.assert_allclose(mean(q), expected, rtol=1e-12)
    midpoint = QuantileBins(8, representation="midpoint_level_linear").fit(x, y, np.ones(200))
    last = np.cumsum([g.size for g in groups]) - 1
    first = np.concatenate([[0], last[:-1]])
    levels = 0.5 * (np.exp(xs[first]) + np.exp(xs[last]))
    np.testing.assert_allclose(midpoint(q), np.interp(np.exp(q[:, 0]), levels, means), rtol=1e-12)
    assert QuantileBins(100, minimum_bin_size=50).fit(x, y, np.ones(200)).count == 4
    with pytest.raises(ValueError):
        QuantileBins(8).fit(x, y, np.arange(200.0))


def test_per_time_means_take_the_fit_of_the_left_node():
    times = jnp.array([0.0, 0.5, 1.0])
    x = RNG.normal(size=(100, 2, 1))
    y = RNG.gamma(2.0, 1.0, (100, 2)) * np.array([1.0, 3.0])
    regression = QuantileBins(5)
    mean = per_time_means(regression, times, x, y, np.ones((100, 2)))
    first = regression.fit(x[:, 0], y[:, 0], np.ones(100))
    second = regression.fit(x[:, 1], y[:, 1], np.ones(100))
    z = jnp.linspace(-1, 1, 7)[:, None]
    np.testing.assert_array_equal(mean(0.2, z), first(z))
    np.testing.assert_array_equal(mean(0.5, z), second(z))
    np.testing.assert_array_equal(mean(1.0, z), second(z))
    assert np.all(np.isnan(mean(1.1, z))) and np.all(np.isnan(mean(-0.1, z)))
    both = mean(jnp.array([0.0, 0.7]), z[:2])
    np.testing.assert_array_equal(both, [first(z[:1])[0], second(z[1:2])[0]])


def test_network_lowers_the_error_and_scales_by_the_offset():
    x = RNG.uniform(-1, 1, (256, 2))
    y = np.exp(0.5 * x[:, 0]) * RNG.gamma(4.0, 0.25, 256)
    network = PositiveMeanNetwork(
        (6,),
        learning_rate=0.02,
        batch_size=32,
        max_epochs=30,
        check_every=5,
        patience=30,
        stream={"seed": 3},
    )
    fitted = network.fit(x, y)
    assert np.mean((fitted(x) - y) ** 2) < np.mean((y.mean() - y) ** 2)
    np.testing.assert_array_equal(network.fit(x, y)(x), fitted(x))
    offset = 0.1 * x[:, 0]
    residual = network.fit(x, y, offset=offset)
    h = residual.unfloored(x, offset) / np.exp(offset)
    np.testing.assert_allclose(residual(x, offset + 0.3), np.exp(offset + 0.3) * h, rtol=1e-12)
    with pytest.raises(ValueError):
        network.fit(x, y, weights=np.arange(256.0) + 1)
    with pytest.raises(ValueError):
        fitted(x, offset)


def test_custom_regression_returns_the_callers_mean():
    def fit(x, y, weights):
        level = jnp.sum(weights * y) / jnp.sum(weights)
        return lambda z: jnp.full(z.shape[:-1], level)

    x, y = RNG.normal(size=(50, 3, 1)), RNG.gamma(2.0, 1.0, (50, 3))
    mean = per_time_means(CustomRegression(fit), jnp.linspace(0, 1, 4), x, y, np.ones((50, 3)))
    np.testing.assert_allclose(mean(0.4, jnp.zeros((2, 1))), y[:, 1].mean(), rtol=1e-14)
