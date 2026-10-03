"""Chebyshev basis, Gamma regression, kernel regression and cubic splines."""

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as numpy_chebyshev

import locsvol  # noqa: F401
from locsvol.regression.chebyshev import ChebyshevTensor, chebyshev
from locsvol.regression.cubic import cubic_minimum, flat_cubic, not_a_knot
from locsvol.regression.gamma import GammaRegression, blocks, loss_and_gradient
from locsvol.regression.kernel import kernel_means

RNG = np.random.default_rng(3)


def test_chebyshev_values_and_gram_from_moments():
    x = jnp.linspace(-1, 1, 9)
    for k in range(6):
        np.testing.assert_allclose(chebyshev(x, 5)[:, k], numpy_chebyshev.chebval(x, [0] * k + [1]))
    basis = ChebyshevTensor((3, 4))
    points = jnp.asarray(RNG.uniform(-1, 1, (500, 2)))
    weights = jnp.asarray(RNG.uniform(0.5, 1.5, 500))
    features = basis(points)
    moments, projection = basis.moments(points, weights, jnp.ones(500))
    np.testing.assert_allclose(
        basis.gram(moments), features.T @ (weights[:, None] * features), atol=1e-11
    )
    np.testing.assert_allclose(projection, features.T @ weights, atol=1e-11)


def test_gamma_regression_recovers_a_log_linear_mean():
    basis = ChebyshevTensor((2, 2))
    x = jnp.asarray(RNG.uniform(-1, 1, (200_000, 2)))
    truth = jnp.zeros(basis.size).at[0].set(0.1).at[1].set(-0.2).at[3].set(0.15)
    offset = 0.05 * x[:, 0]
    mean = jnp.exp(offset + basis(x) @ truth)
    y = jnp.asarray(RNG.gamma(2.0, 1.0, 200_000)) / 2.0 * mean
    regression = GammaRegression(basis, ridge=1e-8, block_rows=50_000)
    alpha = regression.fit(x, y, jnp.ones(200_000), offset)
    np.testing.assert_allclose(alpha, truth, atol=0.02)
    rows = blocks(x, y, jnp.ones(200_000), offset, 50_000)
    _, gradient = loss_and_gradient(alpha, rows, jnp.full(basis.size, 1e-8), basis)
    assert float(jnp.max(jnp.abs(gradient))) <= 1e-8
    np.testing.assert_allclose(
        regression.fit(x, y, jnp.ones(200_000), offset, start=truth), alpha, atol=1e-7
    )


def test_kernel_means_equal_the_direct_formula():
    s = jnp.asarray(RNG.lognormal(0.0, 0.2, 3000))
    y = jnp.asarray(RNG.uniform(0.5, 1.5, 3000))
    queries = jnp.asarray([0.8, 1.0, 1.2])
    for kernel in ("quartic", "gaussian"):
        u = (s[None, :] - queries[:, None]) / 0.1
        k = (
            jnp.where(jnp.abs(u) < 1, (1 - u**2) ** 2, 0.0)
            if kernel == "quartic"
            else jnp.exp(-0.5 * u**2)
        )
        expected = (k @ y) / jnp.sum(k, axis=1)
        np.testing.assert_allclose(
            kernel_means(s, y, queries, jnp.asarray(0.1), kernel, 1000), expected, rtol=1e-12
        )


def test_not_a_knot_spline_reproduces_cubics_and_extends_flat():
    nodes = jnp.linspace(0.5, 1.5, 9)

    def polynomial(x):
        return 1.0 - 0.5 * x + 0.3 * x**2 - 0.2 * x**3

    padded = jnp.concatenate((nodes, jnp.full(3, 1.5)))
    values = jnp.concatenate((polynomial(nodes), jnp.full(3, polynomial(1.5))))
    count = jnp.asarray(9)
    coefficients = not_a_knot(padded, values, count)
    query = jnp.asarray([0.3, 0.55, 0.9, 1.33, 1.7])
    expected = polynomial(jnp.clip(query, 0.5, 1.5))
    np.testing.assert_allclose(
        flat_cubic(padded, values, coefficients, count, query), expected, atol=1e-12
    )
    grid = jnp.linspace(0.5, 1.5, 100001)
    np.testing.assert_allclose(
        cubic_minimum(padded, coefficients, count), jnp.min(polynomial(grid)), atol=1e-9
    )
