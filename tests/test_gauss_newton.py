import jax
import jax.numpy as jnp
import numpy as np
import pytest

import locsvol  # noqa: F401
from locsvol.calibration.gauss_newton import (
    LocalSolver,
    interval_residuals,
    linearize,
    linearized_loss,
    local_loss,
    newton,
    penalized_loss,
    slsqp,
    volatility_coordinates,
    whitening,
)
from locsvol.pricing.black import black_price


def problem(points=True):
    rng = np.random.default_rng(3)
    jacobian = jnp.asarray(rng.standard_normal((12, 4)))
    prices = jnp.asarray(rng.uniform(0.1, 0.2, 12))
    target = prices + jacobian @ jnp.array([0.02, -0.01, 0.03, 0.0])
    width = 0.0 if points else 0.004
    linear = linearize(
        prices, jacobian, target - width, target + width, jnp.ones(12), jnp.full(12, 0.5)
    )
    metric = jnp.asarray(np.diag([1.0, 2.0, 0.5, 1.5]))
    return linear, metric


def test_interval_residuals():
    values = jnp.array([0.9, 1.5, 2.5, 1.2, 7.0])
    lower, upper = jnp.array([1.0, 1.0, 1.0, 1.5, 0.0]), jnp.array([2.0, 2.0, 2.0, 1.5, 1.0])
    weights, scales = jnp.array([4.0, 1.0, 1.0, 1.0, 0.0]), jnp.full(5, 0.5)
    residuals = interval_residuals(values, lower, upper, weights, scales)
    np.testing.assert_allclose(residuals, [-0.4, 0.0, 1.0, -0.6, 0.0], atol=1e-15)


def test_whitening_inverts_the_metric():
    linear, metric = problem()
    w = whitening(metric, linear, 1e-12, 1e-6, jnp.asarray(True))
    np.testing.assert_allclose(w.matrix @ metric @ w.matrix, jnp.eye(4), atol=1e-10)
    weighted = linear.jacobian / 0.5
    leading = jnp.linalg.svd(weighted @ w.matrix, compute_uv=False)[0]
    np.testing.assert_allclose(w.regularization, 1e-6 * leading**2, rtol=1e-12)


def test_newton_solves_the_regularized_least_squares_problem():
    linear, metric = problem()
    solver = LocalSolver(method="newton", tolerance=1e-12, iterations=50)
    loss = local_loss(linear, metric, 1.0, solver)
    a = linear.jacobian / 0.5
    residual = (linear.values - linear.lower) / 0.5
    expected = -jnp.linalg.solve(a.T @ a + loss.regularization * metric, a.T @ residual)
    np.testing.assert_allclose(newton(loss, solver), expected, atol=1e-10)


def test_slsqp_without_constraints_agrees_with_newton():
    linear, metric = problem(points=False)
    loss = local_loss(linear, metric, 1.0, LocalSolver(tolerance=1e-14))
    step = slsqp(loss, None, None, LocalSolver(tolerance=1e-14))
    reference = newton(loss, LocalSolver(method="newton", tolerance=1e-12, iterations=50))
    assert float(penalized_loss(step, loss)) <= float(penalized_loss(reference, loss)) + 1e-9
    assert float(linearized_loss(step, linear)) < float(linearized_loss(jnp.zeros(4), linear))


def test_slsqp_respects_the_trust_region():
    linear, metric = problem()
    radius = 0.005
    solver = LocalSolver()
    loss = local_loss(linear, metric, radius, solver)

    def margins(z):
        delta = loss.transform @ z
        return jnp.atleast_1d(1 - delta @ metric @ delta / radius**2)

    def jacobian(z):
        delta = loss.transform @ z
        return (-2 * (metric @ delta) @ loss.transform / radius**2)[None, :]

    step = slsqp(loss, margins, jacobian, solver)
    assert float(step @ metric @ step) <= radius**2 * (1 + solver.feasibility)
    assert float(linearized_loss(step, linear)) < float(linearized_loss(jnp.zeros(4), linear))


def test_local_loss_rejects_an_indefinite_metric():
    linear, _ = problem()
    with pytest.raises(ValueError):
        local_loss(linear, -jnp.eye(4), 1.0, LocalSolver())


def test_volatility_coordinates_invert_the_black_formula():
    t, k, call = jnp.array([0.5, 1.0]), jnp.array([-0.1, 0.2]), jnp.array([False, True])
    sigma = jnp.array([0.2, 0.3])
    prices = black_price(sigma**2 * t, k, call)
    jacobian = jax.jacfwd(lambda s: black_price(s**2 * t, k, call))(sigma)
    volatility, derivative = volatility_coordinates(prices, jacobian, t, k, call)
    np.testing.assert_allclose(volatility, sigma, rtol=1e-10)
    np.testing.assert_allclose(derivative, jnp.eye(2), atol=1e-9)
