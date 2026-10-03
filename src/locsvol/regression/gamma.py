"""Gamma regression with log link.

The conditional mean of a positive response y given coordinates x is modelled as

    mu(x) = exp(o + b(x)^T alpha)

with a known offset o and basis functions b. The coefficients minimize the Gamma quasi-likelihood
loss with a ridge penalty,

    F(alpha) = sum_n w_n (eta_n + y_n exp(-eta_n)) / sum_n w_n + lambda |alpha|^2 / 2,
    eta_n = o_n + b(x_n)^T alpha.

The second derivative of eta + y exp(-eta) in eta is y exp(-eta) > 0, so for lambda > 0 the
problem is strictly convex and has a unique minimizer. It is found in four steps:

1. the ridge regression of log y - o on b(x), alpha_0 = H^{-1} r, where H = G + lambda I with the
   Gram matrix G = sum_n w_n b(x_n) b(x_n)^T / sum_n w_n and r = sum_n w_n (log y_n - o_n) b(x_n)
   / sum_n w_n; near the minimizer, H is close to the Hessian of F;
2. L-BFGS-B in the coordinates theta = H^{1/2} alpha, in which F is nearly isotropic;
3. L-BFGS-B in alpha;
4. gradient steps alpha - s H^{-1} grad F(alpha) with s = 1, 1/2, 1/4, ..., accepted when F
   decreases by the Armijo condition and max |grad F| decreases, until max |grad F| <= tolerance.

The sums over observations run over blocks of rows, so memory stays bounded for many rows.

Options: the penalty may be a vector lambda_j or a matrix P, with penalty alpha^T P alpha / 2; a
basis other than the Chebyshev one sums G in step 1 from its features; the starting rules
'zeros', 'mean' and 'partition' of gamma_start.py replace steps 1, 2 and 4; a power p != 2
replaces the Gamma loss by the Tweedie loss l_p of tweedie.py.
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy.optimize import minimize

from locsvol.errors import ConvergenceError
from locsvol.regression.gamma_start import conditioned, dense_ridge_start, rule_start
from locsvol.regression.tweedie import tweedie_loss

ARMIJO = 1e-4
LBFGSB = dict(maxls=40, maxcor=20, ftol=1e-14)  # line-search trials, memory, relative decrease
LINE_SEARCH = 30  # step halvings in step 4


class Rows(NamedTuple):
    """Observations in blocks of equal size; padding rows have weight zero."""

    x: jax.Array
    y: jax.Array
    offset: jax.Array
    weights: jax.Array


def blocks(x, y, weights, offset, size):
    """Rows of (x, y, w, o) in blocks of at most size rows."""
    x, y, w = (jnp.asarray(v, dtype=jnp.float64) for v in (x, y, weights))
    o = jnp.broadcast_to(jnp.asarray(offset, dtype=jnp.float64), y.shape)
    size = min(size, y.size)
    padding = (-y.size) % size
    return Rows(
        jnp.pad(x, ((0, padding), (0, 0))).reshape(-1, size, x.shape[1]),
        jnp.pad(y, (0, padding), constant_values=1.0).reshape(-1, size),
        jnp.pad(o, (0, padding)).reshape(-1, size),
        jnp.pad(w, (0, padding)).reshape(-1, size),
    )


def _block_loss(alpha, rows, basis, power):
    x, y, o, w = rows
    active = w != 0
    eta = o + basis(jnp.where(active[:, None], x, 0.0)) @ alpha
    return jnp.sum(w * tweedie_loss(jnp.where(active, y, 1.0), jnp.where(active, eta, 0.0), power))


@partial(jax.jit, static_argnames=("basis", "power"))
def loss_and_gradient(alpha, rows, penalty, basis, power=2.0):
    """F(alpha) and grad F(alpha) for the penalty, a vector (ridge weights) or a matrix."""

    def add(total, block):
        value, gradient = jax.value_and_grad(_block_loss)(alpha, block, basis, power)
        return (total[0] + value, total[1] + gradient), None

    def quadratic(a):
        return 0.5 * jnp.dot(a, penalty @ a if penalty.ndim == 2 else penalty * a)

    (value, gradient), _ = lax.scan(add, (jnp.zeros(()), jnp.zeros_like(alpha)), rows)
    mass = jnp.sum(rows.weights)
    ridge, ridge_gradient = jax.value_and_grad(quadratic)(alpha)
    return value / mass + ridge, gradient / mass + ridge_gradient


@partial(jax.jit, static_argnames=("basis", "power"))
def _transformed(theta, transform, rows, penalty, basis, power):
    value, gradient = loss_and_gradient(transform @ theta, rows, penalty, basis, power)
    return value, transform.T @ gradient


@partial(jax.jit, static_argnames="basis")
def ridge_start(rows, penalty, basis):
    """Step 1: alpha_0 = H^{-1} r, and H^{1/2} and H^{-1/2} for the change of coordinates.

    H gets the smallest shift that makes its eigenvalues at least machine precision times its
    largest one.
    """

    def add(total, block):
        x, y, o, w = block
        active = w != 0
        z = jnp.log(jnp.where(active, y, 1.0)) - jnp.where(active, o, 0.0)
        moments, projection = basis.moments(jnp.where(active[:, None], x, 0.0), w, z)
        return (total[0] + moments, total[1] + projection.reshape(total[1].shape)), None

    p, q = basis.degrees
    zero = (jnp.zeros((2 * p + 1, 2 * q + 1)), jnp.zeros((p + 1, q + 1)))
    (moments, projection), _ = lax.scan(add, zero, rows)
    mass = jnp.sum(rows.weights)
    return conditioned(basis.gram(moments) / mass, projection.reshape(basis.size) / mass, penalty)


class _BudgetExhausted(Exception):
    pass


@dataclass(frozen=True)
class GammaRegression:
    """Gamma regression with log link and ridge penalty on a basis; see the module docstring.

    ridge is lambda, a vector of lambda_j or a penalty matrix P; initialization is a rule of
    gamma_start.RULES, intercept the index of the constant basis function for the rule 'mean'.
    """

    basis: object
    ridge: object
    tolerance: float = 1e-8  # on max |grad F|
    max_iterations: int = 300  # L-BFGS-B iterations and gradient steps together
    max_evaluations: int = 600  # evaluations of F and grad F together
    block_rows: int = 32768
    power: float = 2.0
    initialization: str = "ridge"
    intercept: int | None = None

    def fit(self, x, y, weights, offset=0.0, start=None):
        """The coefficients alpha for rows x (n, d), responses y > 0, weights w >= 0, offsets o.

        start replaces alpha_0 of step 1, for instance by the previous fit of a fixed-point
        iteration.
        """
        rows = blocks(x, y, weights, offset, self.block_rows)
        penalty = jnp.asarray(self.ridge, dtype=jnp.float64)
        penalty = penalty if penalty.ndim == 2 else jnp.full(self.basis.size, penalty)
        whitened = self.initialization == "ridge"
        if whitened:
            first = ridge_start if hasattr(self.basis, "moments") else dense_ridge_start
            alpha, root, inverse_root = first(rows, penalty, self.basis)
        else:
            alpha = rule_start(self.initialization, rows, self.basis, self.intercept)
            root = inverse_root = jnp.eye(self.basis.size)
        if start is not None:
            alpha = jnp.asarray(start, dtype=jnp.float64)
        evaluations = 0

        def lbfgsb(transform, theta, iterations):
            def objective(theta):
                nonlocal evaluations
                if evaluations >= self.max_evaluations:
                    raise _BudgetExhausted
                evaluations += 1
                value, gradient = _transformed(
                    jnp.asarray(theta), transform, rows, penalty, self.basis, self.power
                )
                return float(value), np.asarray(gradient)

            options = dict(
                LBFGSB,
                maxiter=iterations,
                maxfun=self.max_evaluations - evaluations,
                gtol=self.tolerance,
            )
            return minimize(
                objective, np.asarray(theta), jac=True, method="L-BFGS-B", options=options
            )

        used, exhausted = 0, False
        try:
            if whitened and self.max_iterations > 1:
                result = lbfgsb(inverse_root, root @ alpha, self.max_iterations - 1)
                used = min(result.nit, self.max_iterations - 1)
                proposal = inverse_root @ jnp.asarray(result.x)
                value, _ = loss_and_gradient(proposal, rows, penalty, self.basis, self.power)
                if bool(jnp.all(jnp.isfinite(proposal)) & jnp.isfinite(value)):
                    alpha = proposal
            identity = jnp.eye(self.basis.size)
            result = lbfgsb(identity, alpha, self.max_iterations - used)
            used += min(result.nit, self.max_iterations - used)
            alpha = jnp.asarray(result.x)
        except _BudgetExhausted:
            exhausted = True
        value, gradient = loss_and_gradient(alpha, rows, penalty, self.basis, self.power)
        finite = bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient)))
        if whitened and not exhausted and finite:
            alpha, gradient = self._gradient_steps(
                alpha,
                value,
                gradient,
                inverse_root,
                rows,
                penalty,
                self.max_iterations - used,
                self.max_evaluations - evaluations,
            )
        largest = float(jnp.max(jnp.abs(gradient)))
        if not largest <= self.tolerance:
            raise ConvergenceError(
                f"Gamma regression: max |grad F| = {largest:.2e} > {self.tolerance:.0e}"
            )
        return alpha

    def _gradient_steps(self, alpha, value, gradient, inverse_root, rows, penalty, steps, budget):
        """Step 4 within the remaining iterations and evaluations."""

        def norm(g):
            return jnp.max(jnp.abs(g))

        taken = used = 0
        while taken < steps and used < budget and norm(gradient) > self.tolerance:
            direction = -inverse_root @ (inverse_root @ gradient)
            slope = jnp.dot(gradient, direction)
            if not bool(jnp.isfinite(slope) & (slope < 0)):
                break
            step, accepted, trials = 1.0, False, 0
            while not accepted and trials < LINE_SEARCH and used + trials < budget:
                candidate = alpha + step * direction
                trial_value, trial_gradient = loss_and_gradient(
                    candidate, rows, penalty, self.basis, self.power
                )
                accepted = bool(
                    jnp.isfinite(trial_value)
                    & jnp.all(jnp.isfinite(trial_gradient))
                    & (trial_value <= value + ARMIJO * step * slope)
                    & (norm(trial_gradient) < norm(gradient))
                )
                trials += 1
                step *= 0.5
            used += trials
            if not accepted:
                break
            alpha, value, gradient = candidate, trial_value, trial_gradient
            taken += 1
        return alpha, gradient
