"""Starting rules, penalty matrices and Tweedie powers of the regression with log link."""

import jax.numpy as jnp
import numpy as np
import pytest

import locsvol  # noqa: F401
from locsvol.regression.gamma import GammaRegression, blocks
from locsvol.regression.gamma_start import dense_ridge_start
from locsvol.regression.mixed import TimeSplineChebyshev
from locsvol.regression.polynomials import Powers
from locsvol.regression.splines import GammaSplines, spline_penalty

RNG = np.random.default_rng(13)
X = np.stack([RNG.uniform(-1, 1, 3000), np.clip(RNG.normal(0, 0.4, 3000), -1, 1)], axis=1)
OFFSET = 0.1 * np.sin(2 * X[:, 0])
MEAN = np.exp(OFFSET + 0.2 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 0] * X[:, 1])
Y = RNG.gamma(2.0, MEAN / 2.0)
W = RNG.uniform(0.5, 1.5, 3000)
POWERS = Powers(((0, 0), (1, 0), (0, 1), (1, 1), (0, 2)))


def gradient(alpha, basis, penalty, y=Y, power=2.0):
    """grad F = sum w mu^{1-p} (mu - y) b / sum w + P alpha."""
    features = np.asarray(basis(X))
    mu = np.exp(OFFSET + features @ alpha)
    return features.T @ (W * mu ** (1 - power) * (mu - y)) / W.sum() + penalty @ alpha


def test_dense_ridge_start_solves_the_ridge_regression_of_log_y():
    penalty = np.full(POWERS.size, 1e-3)
    start, root, inverse_root = dense_ridge_start(
        blocks(X, Y, W, OFFSET, 700), jnp.asarray(penalty), POWERS
    )
    features = np.asarray(POWERS(X))
    gram = features.T @ (W[:, None] * features) / W.sum() + np.diag(penalty)
    right = features.T @ (W * (np.log(Y) - OFFSET)) / W.sum()
    np.testing.assert_allclose(start, np.linalg.solve(gram, right), rtol=1e-10)
    np.testing.assert_allclose(root @ root, gram, atol=1e-13)
    np.testing.assert_allclose(inverse_root @ root, np.eye(POWERS.size), atol=1e-12)


@pytest.mark.parametrize(
    "basis, rules",
    [
        (POWERS, ("ridge", "zeros", "mean")),
        (TimeSplineChebyshev((-1, -1, -1, 0, 1, 1, 1), 2, 3), ("ridge", "zeros", "partition")),
    ],
)
def test_every_starting_rule_finds_the_minimizer(basis, rules):
    fits = []
    for rule in rules:
        regression = GammaRegression(basis, 1e-4, initialization=rule, intercept=0)
        alpha = regression.fit(X, Y, W, OFFSET)
        assert np.max(np.abs(gradient(alpha, basis, 1e-4 * np.eye(basis.size)))) < 1e-8
        fits.append(alpha)
    np.testing.assert_allclose(fits[1:], [fits[0]] * (len(fits) - 1), atol=1e-6)


@pytest.mark.parametrize("power", [1.0, 1.5, 3.0])
def test_tweedie_powers_solve_their_first_order_condition(power):
    y = RNG.poisson(MEAN).astype(float) if power == 1.0 else Y
    regression = GammaRegression(POWERS, 1e-5, power=power, initialization="mean", intercept=0)
    alpha = regression.fit(X, y, W, OFFSET)
    penalty = 1e-5 * np.eye(POWERS.size)
    assert np.max(np.abs(gradient(alpha, POWERS, penalty, y, power))) < 1e-8


def test_gamma_splines_with_second_difference_penalty():
    mean = GammaSplines((5, 4), (1e-3, 1e-2)).fit(X, Y, W, OFFSET)
    penalty = np.asarray(spline_penalty(mean.basis.sizes, (1e-3, 1e-2)))
    assert np.max(np.abs(gradient(np.asarray(mean.coefficients), mean.basis, penalty))) < 1e-8
    np.testing.assert_allclose(
        mean(X[:5], OFFSET[:5]), np.exp(OFFSET[:5] + mean.basis(X[:5]) @ mean.coefficients)
    )
