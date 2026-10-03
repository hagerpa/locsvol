"""Regularized Gauss-Newton steps in a trust region, for losses on bid-ask intervals.

The loss of model prices P(beta) on quotes q with intervals [l_q, u_q], weights w_q and scales
v_q (for instance the Black vegas) is the scaled distance to the intervals,

    F(beta) = sum_q r_q(P_q(beta))^2,    r_q(p) = sqrt(w_q) dist(p, [l_q, u_q]) / v_q,

with the signed distance dist(p, [l, u]) = p - u above the interval, p - l below and 0 inside;
a point quote has l = u. For a loss in implied volatility, the prices are replaced by their
Black implied volatilities and J by J / vega. A step linearizes the prices,
P(beta + delta) ~ P + J delta, and minimizes the local loss

    G(delta) = (sum_q r_q(P_q + J_q delta)^2 + lambda delta^T M delta) / c

in a trust region delta^T M delta <= rho^2, where M is a metric of the parameters and c a
constant that scales the local loss for the solver. The penalty weight lambda = lambda_0
s_max(A Q)^2 follows the largest singular value of the weighted Jacobian
A = diag(sqrt(w) / v) J in the coordinates z with delta = rho Q z, where
Q = (M + epsilon mu_max I)^{-1/2} is taken from the eigendecomposition M = V diag(mu) V^T with
negative eigenvalues set to zero; in these coordinates the trust region is about the unit
ball. Since r_q is piecewise linear, G is convex and piecewise quadratic.

The local problem is solved by SLSQP (Kraft, DFVLR-FB 88-28, 1988) under the trust region and
the further inequality constraints of the caller, or without any constraint by a semismooth
Newton iteration: on the piece of G selected by the quotes currently outside their intervals,
the Hessian is exact, so a full step reaches the minimizer of that piece, and Armijo
backtracking keeps the iteration monotone across pieces.
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from locsvol.errors import ConvergenceError
from locsvol.pricing.black import black_vega, implied_volatility


class Linearization(NamedTuple):
    """Prices (quotes,), their Jacobian (quotes, parameters), intervals, weights and scales."""

    values: jax.Array
    jacobian: jax.Array
    lower: jax.Array
    upper: jax.Array
    weights: jax.Array
    scales: jax.Array


@dataclass(frozen=True)
class LocalSolver:
    """Settings of the local problem: regularization lambda_0, ridge epsilon and divisor c of
    the module docstring (lambda = lambda_0 without spectral scaling), the solver, "slsqp" or
    "newton", with its iterations and tolerance (SLSQP's ftol, or the Newton tolerance on the
    gradient relative to its start), and the tolerance on the constraint margins."""

    method: str = "slsqp"
    iterations: int = 1500
    tolerance: float = 1e-8
    feasibility: float = 1e-7
    divisor: float = 0.0015**2
    regularization: float = 1e-6
    spectral: bool = True
    ridge: float = 1e-12


@jax.jit
def interval_residuals(values, lower, upper, weights, scales):
    """r_q = sqrt(w_q) dist(p_q, [l_q, u_q]) / v_q; quotes of weight zero give zero."""
    active = weights > 0
    y = jnp.where(active, values, lower)
    excess = jnp.where(y < lower, y - lower, jnp.where(y > upper, y - upper, 0.0))
    excess = jnp.where(lower == upper, y - lower, excess)
    return jnp.where(active & ~jnp.isfinite(y), jnp.nan, jnp.sqrt(weights) * excess / scales)


def volatility_coordinates(prices, jacobian, maturities, log_strikes, is_call):
    """Black implied volatilities sigma of the prices and their derivatives J / vega(sigma)."""
    sigma = implied_volatility(prices, maturities, log_strikes, is_call)
    return sigma, (1 / black_vega(sigma, maturities, log_strikes))[:, None] * jacobian


def linearize(prices, jacobian, lower, upper, weights, scales):
    """The linearization, with the rows of quotes of weight zero set to their lower bound."""
    active = jnp.asarray(weights) > 0
    result = Linearization(
        jnp.where(active, prices, lower),
        jnp.where(active[:, None], jacobian, 0.0),
        *(jnp.asarray(v, dtype=jnp.float64) for v in (lower, upper, weights, scales)),
    )
    if not bool(jnp.all(jnp.isfinite(result.values)) & jnp.all(jnp.isfinite(result.jacobian))):
        raise ValueError("the prices or their derivatives are not finite")
    return result


@jax.jit
def linearized_loss(delta, linearization):
    """sum_q r_q(P_q + J_q delta)^2."""
    a = linearization
    r = interval_residuals(a.values + a.jacobian @ delta, a.lower, a.upper, a.weights, a.scales)
    return jnp.vdot(r, r)


class Whitening(NamedTuple):
    """Q of the module docstring, the eigenvalues of M and the penalty weight lambda."""

    matrix: jax.Array
    eigenvalues: jax.Array
    regularization: jax.Array


@jax.jit
def whitening(metric, linearization, ridge, strength, spectral):
    """Q = V diag(max(mu, 0) + ridge mu_max)^{-1/2} V^T and lambda = strength s_max(A Q)^2,
    or lambda = strength without spectral scaling."""
    eigenvalues, vectors = jnp.linalg.eigh(metric)
    values = jnp.maximum(eigenvalues, 0) + ridge * eigenvalues[-1]
    q = (vectors / jnp.sqrt(values)) @ vectors.T
    a = linearization
    weighted = (jnp.sqrt(a.weights) / a.scales)[:, None] * a.jacobian
    leading = jnp.linalg.svd(weighted @ q, compute_uv=False)[0]
    return Whitening(q, eigenvalues, strength * jnp.where(spectral, leading**2, 1.0))


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["linearization", "metric", "transform", "regularization", "divisor"],
    meta_fields=[],
)
@dataclass(frozen=True)
class LocalLoss:
    """G(rho Q z) of the module docstring as a function of z; transform is rho Q."""

    linearization: Linearization
    metric: jax.Array
    transform: jax.Array
    regularization: jax.Array
    divisor: float


def local_loss(linearization, metric, radius, solver):
    """The local loss with its whitening, for the trust region radius."""
    w = whitening(
        metric, linearization, solver.ridge, solver.regularization, jnp.asarray(solver.spectral)
    )
    largest, smallest = float(w.eigenvalues[-1]), float(w.eigenvalues[0])
    if not (largest > 0 and smallest >= -solver.ridge * largest):
        raise ValueError("the metric must be positive semidefinite and nonzero")
    if not bool(jnp.all(jnp.isfinite(w.matrix)) & jnp.isfinite(w.regularization)):
        raise ValueError("the metric has no finite whitening")
    return LocalLoss(linearization, metric, radius * w.matrix, w.regularization, solver.divisor)


@jax.jit
def penalized_loss(delta, loss):
    """G(delta) = (sum_q r_q(P_q + J_q delta)^2 + lambda delta^T M delta) / c."""
    penalty = loss.regularization * (delta @ loss.metric @ delta)
    return (linearized_loss(delta, loss.linearization) + penalty) / loss.divisor


@jax.jit
def _loss_in_coordinates(z, loss):
    return penalized_loss(loss.transform @ z, loss)


_loss_and_gradient = jax.jit(jax.value_and_grad(_loss_in_coordinates))


def slsqp(loss, margins, margin_jacobian, solver):
    """The step delta = rho Q z minimizing the local loss under margins(z) >= 0, by SLSQP.

    margins and margin_jacobian map z to the constraint values and their Jacobian, or are None
    without constraints; the margin tolerance of the solver applies to both the start z = 0
    and the solution.
    """

    def objective(z):
        value, gradient = _loss_and_gradient(jnp.asarray(z), loss)
        return float(value), np.asarray(gradient)

    def constraints(z):
        return np.zeros(0) if margins is None else np.asarray(margins(jnp.asarray(z)))

    def jacobian(z):
        return np.asarray(margin_jacobian(jnp.asarray(z)))

    zero = np.zeros(loss.transform.shape[1])
    start = constraints(zero)
    if np.any(~np.isfinite(start) | (start < -solver.feasibility)):
        raise ValueError("the current parameters violate the constraints")
    result = minimize(
        objective,
        zero,
        jac=True,
        method="SLSQP",
        constraints=()
        if margins is None
        else {"type": "ineq", "fun": constraints, "jac": jacobian},
        options={"maxiter": solver.iterations, "ftol": solver.tolerance, "disp": False},
    )
    reached = constraints(result.x)
    if (
        not result.success
        or not np.all(np.isfinite(result.x))
        or np.any(~np.isfinite(reached) | (reached < -solver.feasibility))
        or not np.isfinite(result.fun)
    ):
        raise ConvergenceError(f"SLSQP: status {result.status}, {result.message}")
    return loss.transform @ jnp.asarray(result.x)


@partial(jax.jit, static_argnames="iterations")
def _newton_coordinates(loss, tolerance, iterations):
    a, t = loss.linearization, loss.transform
    weighted = (jnp.sqrt(a.weights) / a.scales)[:, None] * (a.jacobian @ t)
    penalty = t.T @ (loss.regularization * loss.metric) @ t
    point = a.lower == a.upper

    def outside(z):
        values = a.values + a.jacobian @ (t @ z)
        return (interval_residuals(values, a.lower, a.upper, a.weights, a.scales) != 0) | point

    def direction(z, gradient):
        rows = jnp.where(outside(z)[:, None], weighted, 0.0)
        hessian = 2 * (rows.T @ rows + penalty) / loss.divisor
        # Without penalty, an active set of low rank leaves flat directions; the step of
        # least norm moves along the curved ones only.
        values, vectors = jnp.linalg.eigh(hessian)
        kept = values > 1e-12 * jnp.maximum(values[-1], 0.0)
        inverse = jnp.where(kept, 1 / jnp.where(kept, values, 1.0), 0.0)
        return -vectors @ (inverse * (vectors.T @ gradient))

    z = jnp.zeros(t.shape[1])
    value, gradient = _loss_and_gradient(z, loss)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(gradient)))

    def unresolved(carry):
        _, _, gradient, count, stalled = carry
        return (jnp.max(jnp.abs(gradient)) > tolerance * scale) & (count < iterations) & ~stalled

    def iterate(carry):
        z, value, gradient, count, _ = carry
        step = direction(z, gradient)
        slope = gradient @ step

        def insufficient(inner):
            length, trial, halvings = inner
            return (trial > value + 1e-4 * length * slope) & (halvings < 40)

        def halve(inner):
            length, _, halvings = inner
            length = 0.5 * length
            return length, _loss_in_coordinates(z + length * step, loss), halvings + 1

        length, trial, _ = jax.lax.while_loop(
            insufficient, halve, (jnp.asarray(1.0), _loss_in_coordinates(z + step, loss), 0)
        )
        accepted = (trial <= value + 1e-4 * length * slope) & jnp.all(jnp.isfinite(step))
        following = jnp.where(accepted, z + length * step, z)
        value, gradient = _loss_and_gradient(following, loss)
        return following, value, gradient, count + 1, ~accepted

    z, value, gradient, _, _ = jax.lax.while_loop(
        unresolved, iterate, (z, value, gradient, 0, jnp.asarray(False))
    )
    converged = jnp.max(jnp.abs(gradient)) <= tolerance * scale
    return z, converged & jnp.all(jnp.isfinite(z)) & jnp.isfinite(value)


def newton(loss, solver):
    """The step delta = rho Q z minimizing the local loss without constraints."""
    z, converged = _newton_coordinates(loss, solver.tolerance, solver.iterations)
    if not bool(converged):
        raise ConvergenceError("the Newton iteration of the local loss did not converge")
    return loss.transform @ z
