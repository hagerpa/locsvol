"""Polynomial, B-spline, mixed and custom bases."""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import chebyshev as numpy_chebyshev
from scipy.interpolate import BSpline

import locsvol  # noqa: F401
from locsvol.regression.custom import CustomBasis
from locsvol.regression.mixed import TimeSplineChebyshev
from locsvol.regression.polynomials import Chebyshev, Identity, Powers
from locsvol.regression.splines import bsplines, spline_basis, spline_penalty

RNG = np.random.default_rng(5)


def test_power_bases_follow_their_formulas():
    x = RNG.uniform(-3, 3, (50, 2))
    powers = ((0, 0), (1, 0), (0, 2), (2, 1))
    monomials = np.stack([x[:, 0] ** a * x[:, 1] ** b for a, b in powers], axis=1)
    np.testing.assert_allclose(Powers(powers)(x), monomials, rtol=1e-14)
    np.testing.assert_allclose(Powers(powers, bounds=(-1.0, 2.0))(x), np.clip(monomials, -1, 2))
    rational = monomials / (1 + np.sum(np.abs(x) ** 3, axis=1))[:, None]
    np.testing.assert_allclose(Powers(powers, rational_degree=3)(x), rational, rtol=1e-13)
    np.testing.assert_array_equal(Identity(2)(x), x)
    with pytest.raises(ValueError):
        Powers(powers, rational_degree=2)
    u = np.linspace(-1, 1, 9)[:, None]
    values = np.stack([numpy_chebyshev.chebval(u[:, 0], [0] * j + [1]) for j in range(5)], axis=1)
    np.testing.assert_allclose(Chebyshev(4)(u), values, atol=1e-14)


def test_bsplines_agree_with_scipy_and_sum_to_one():
    knots = np.array([-1.6, -1.2, -0.8, -0.5, -0.1, 0.3, 0.6, 1.0, 1.4, 1.8])
    x = np.linspace(knots[3], knots[-4], 101)
    values = np.asarray(bsplines(jnp.asarray(x), jnp.asarray(knots), 3))
    np.testing.assert_allclose(values, BSpline.design_matrix(x, knots, 3).toarray(), atol=1e-14)
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-14)


def test_knots_follow_the_quantiles_and_continue_the_outer_spacing():
    x = RNG.normal(size=(400, 2))
    basis = spline_basis(x, (5, 4), degree=2)
    for column, knots, count in zip(x.T, basis.knots, (5, 4), strict=True):
        base = np.quantile(column, np.linspace(0, 1, count))
        first, last = base[1] - base[0], base[-1] - base[-2]
        expected = np.concatenate(
            [base[0] - first * np.array([2, 1]), base, base[-1] + last * np.array([1, 2])]
        )
        np.testing.assert_allclose(knots, expected, rtol=1e-14)
    assert basis.sizes == (6, 5) and basis(x).shape == (400, 30)
    np.testing.assert_allclose(basis(x) @ basis.unit, 1.0, atol=1e-13)
    uniform = spline_basis(x, (3, 3), degree=1, rule="uniform")
    np.testing.assert_allclose(uniform.knots[0], [-2.0, -1.0, 0.0, 1.0, 2.0])
    collapsed = spline_basis(np.stack([np.zeros(400), x[:, 1]], axis=1), (5, 4))
    assert collapsed.knots[0] == () and collapsed.sizes[0] == 1


def test_extension_beyond_the_base_interval():
    x = RNG.normal(size=(300, 1))
    constant = spline_basis(x, (6,))
    linear = spline_basis(x, (6,), extrapolation="linear")
    upper = constant.knots[0][-4]
    inside, outside = jnp.asarray([[upper], [upper - 1e-6]]), jnp.asarray([[upper + 0.5]])
    np.testing.assert_allclose(constant(outside), constant(inside[:1]), atol=1e-15)
    end, before = linear(inside)
    slope = (end - before) / 1e-6
    np.testing.assert_allclose(linear(outside)[0], end + 0.5 * slope, atol=1e-5)


def test_penalty_sums_squared_second_differences():
    alpha = RNG.normal(size=35)
    grid = alpha.reshape(5, 7)
    expected = 0.3 * np.sum(np.diff(grid, 2, axis=0) ** 2)
    expected += 2.0 * np.sum(np.diff(grid, 2, axis=1) ** 2)
    np.testing.assert_allclose(alpha @ spline_penalty((5, 7), (0.3, 2.0)) @ alpha, expected)
    np.testing.assert_array_equal(spline_penalty((2,), (1.0,)), np.zeros((2, 2)))


def test_mixed_basis_multiplies_splines_and_polynomials():
    knots = (-1, -1, -1, -1, -0.5, 0.2, 1, 1, 1, 1)
    basis = TimeSplineChebyshev(knots, 3, 4)
    x = RNG.uniform(-1, 1, (60, 2))
    time = BSpline.design_matrix(x[:, 0], np.asarray(knots, dtype=float), 3).toarray()
    state = np.stack([numpy_chebyshev.chebval(x[:, 1], [0] * j + [1]) for j in range(5)], axis=1)
    expected = (time[:, :, None] * state[:, None, :]).reshape(60, -1)
    np.testing.assert_allclose(basis(x), expected, atol=1e-13)
    np.testing.assert_allclose(basis(x) @ basis.unit, 1.0, atol=1e-13)
    rejecting = TimeSplineChebyshev(
        knots, 3, 4, time_extrapolation="reject", state_extrapolation="reject"
    )
    values = rejecting(jnp.array([[1.5, 0.0], [0.0, -1.5], [0.0, 0.0]]))
    assert np.all(np.isnan(values[:2])) and np.all(np.isfinite(values[2]))
    with pytest.raises(ValueError):
        TimeSplineChebyshev((-1, -1, -0.5, 0.2, 1, 1), 3, 4)


def test_custom_basis_checks_its_size():
    def features(z):
        return jnp.stack([jnp.ones(z.shape[:-1]), jnp.sin(z[..., 0])], axis=-1)

    assert CustomBasis(features, 2)(jnp.zeros((3, 1))).shape == (3, 2)
    with pytest.raises(ValueError):
        CustomBasis(features, 3)(jnp.zeros((3, 1)))
