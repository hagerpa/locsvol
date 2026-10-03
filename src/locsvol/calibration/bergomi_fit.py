"""The two-factor Bergomi backbone fitted to the quotes of an earlier day.

The pure Bergomi model has the leverage factor L(t, x) = sqrt(xi(t)) of the backbone's forward
variance curve, so that dS_t / S_t = sqrt(xi(t) Y_t) dW_t with the normalized variance factor Y
of backbones.bergomi. Selected parameters p, by default (nu, theta, rho_1, rho_2), are fitted to
the quotes by regularized Gauss-Newton steps in a trust region (calibration.gauss_newton) in
the coordinates x = (p - p_0) / s around the starting point p_0 with scales s, with the identity
as metric. The step delta from x satisfies

    |delta| <= rho_i,    |x + delta| <= R,    l <= p_0 + s (x + delta) <= u,

and keeps the correlation matrix of (W, Z^1, Z^2) positive definite with a margin epsilon: its
principal minors 1 - rho_1^2, 1 - rho_2^2, 1 - rho_Z^2 and its determinant
1 + 2 rho_1 rho_2 rho_Z - rho_1^2 - rho_2^2 - rho_Z^2 are at least epsilon.

Alternatively, the loss itself is minimized under the bounds and the correlation constraints by
the interior point method with trust regions of Byrd, Hribar and Nocedal (SIAM J. Optim. 9, 1999)
and quasi-Newton (BFGS) Hessians, with the gradient 2 J^T diag(w / v^2) r of the loss from the
price derivatives J and the residuals r.

The prices are Monte Carlo prices on fixed Brownian increments, independent replications
averaged, and their derivatives in x come from forward-mode differentiation of the simulation
of the variance factor and the spot, in chunks of paths.
"""

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import BFGS, Bounds, NonlinearConstraint, minimize

from locsvol.backbones.bergomi import ForwardVariance
from locsvol.calibration.gauss_newton import (
    LocalSolver,
    linearize,
    linearized_loss,
    local_loss,
    slsqp,
    volatility_coordinates,
)
from locsvol.errors import ConvergenceError
from locsvol.functions import compiled
from locsvol.pricing.monte_carlo import chunks, maturity_nodes, payoffs
from locsvol.simulation.euler import log_spot

PARAMETERS = ("nu", "theta", "rho1", "rho2")  # fitted in the paper, with these bounds and scales
LOWER = (0.05, 0.05, -0.95, -0.95)
UPPER = (4.0, 0.95, 0.95, 0.95)
SCALES = (1.0, 0.5, 0.5, 0.5)
SOLVER = LocalSolver(tolerance=1e-10)
INTERIOR_POINT = dict(  # options of scipy's trust-constr
    maxiter=150,
    gtol=1e-9,
    xtol=1e-11,
    barrier_tol=1e-10,
    initial_tr_radius=0.2,
    initial_constr_penalty=1.0,
    initial_barrier_parameter=1e-4,
    initial_barrier_tolerance=1e-4,
)


@partial(jax.tree_util.register_dataclass, data_fields=["forward_variance"], meta_fields=[])
@dataclass(frozen=True)
class ForwardVolatility:
    """L(t, x) = sqrt(xi(t)), the leverage factor of the pure Bergomi model."""

    forward_variance: ForwardVariance

    def __call__(self, t, x):
        return jnp.sqrt(self.forward_variance(t)) + jnp.zeros_like(x)


def displaced(template, x, scales, names):
    """The backbone with the parameters p_0 + s x."""
    updates = {name: getattr(template, name) + scales[i] * x[i] for i, name in enumerate(names)}
    return replace(template, **updates)


def correlation_minors(backbone):
    """1 - rho_1^2, 1 - rho_2^2, 1 - rho_Z^2 and the determinant of the correlation matrix."""
    a, b, c = backbone.rho1, backbone.rho2, backbone.rho_z
    return jnp.stack((1 - a * a, 1 - b * b, 1 - c * c, 1 + 2 * a * b * c - a * a - b * b - c * c))


@partial(jax.jit, static_argnames="names")
def _chunk_prices(x, template, scales, names, leverage, times, increments, nodes, k, is_call):
    def prices(x):
        backbone = displaced(template, x, scales, names)
        spot = log_spot(leverage, times, backbone.variance(times, increments), increments[:, :, 0])
        return jnp.mean(payoffs(spot[:, nodes], k, is_call), axis=0)

    value, tangents = jax.vmap(lambda v: jax.jvp(prices, (x,), (v,)), out_axes=(None, 0))(
        jnp.eye(x.size)
    )
    return value, tangents.T


def bergomi_prices(
    template,
    x,
    times,
    increments,
    maturities,
    log_strikes,
    is_call,
    leverage=None,
    names=PARAMETERS,
    scales=SCALES,
):
    """Prices (quotes,) of the model with the backbone p_0 + s x and their derivatives in x
    (quotes, size), averaged over the chunks of all replications.

    increments are the replications of Brownian increments (paths, steps, 3) on the grid times;
    the leverage factor is sqrt(xi(t)) of the template unless given.
    """
    if any(replication.shape[1:] != (len(times) - 1, 3) for replication in increments):
        raise ValueError("the increments must have shape (paths, steps of times, 3)")
    leverage = ForwardVolatility(template.forward_variance) if leverage is None else leverage
    leverage = compiled(leverage)
    nodes = jnp.asarray(maturity_nodes(times, maturities))
    k, call, s = jnp.asarray(log_strikes), jnp.asarray(is_call), jnp.asarray(scales)
    prices, jacobian = jnp.zeros(k.size), jnp.zeros((k.size, x.size))
    for replication in increments:
        count = replication.shape[0]
        for part in chunks(count):
            weight = (part.stop - part.start) / count / len(increments)
            value, derivative = _chunk_prices(
                x, template, s, names, leverage, times, replication[part], nodes, k, call
            )
            prices = prices + weight * value
            jacobian = jacobian + weight * derivative
    return prices, jacobian


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "template",
        "scales",
        "lower",
        "upper",
        "margin",
        "displacement",
        "transform",
        "radius",
        "cumulative_radius",
    ],
    meta_fields=["names"],
)
@dataclass(frozen=True)
class Constraints:
    """The constraints of one step on z, with delta = transform z from x = displacement; lower
    and upper are the bounds in the coordinates x."""

    template: object
    scales: jax.Array
    lower: jax.Array
    upper: jax.Array
    margin: float
    displacement: jax.Array
    transform: jax.Array
    radius: float
    cumulative_radius: float | None
    names: tuple


@jax.jit
def margins(z, constraints):
    """Margins, nonnegative where the constraints hold: trust region, lower and upper bounds,
    correlation minors and, if given, the cumulative radius."""
    c = constraints
    delta = c.transform @ z
    candidate = c.displacement + delta
    width = c.upper - c.lower
    minors = correlation_minors(displaced(c.template, candidate, c.scales, c.names))
    parts = [
        jnp.atleast_1d(1 - jnp.vdot(delta, delta) / c.radius**2),
        (candidate - c.lower) / width,
        (c.upper - candidate) / width,
        minors - c.margin,
    ]
    if c.cumulative_radius is not None:
        parts.append(jnp.atleast_1d(1 - jnp.vdot(candidate, candidate) / c.cumulative_radius**2))
    return jnp.concatenate(parts)


margin_jacobian = jax.jit(jax.jacfwd(margins))


def _admissible(backbone, names, lower, upper, margin):
    p = jnp.asarray([getattr(backbone, name) for name in names])
    return bool(
        jnp.all(jnp.isfinite(p))
        & jnp.all((p >= jnp.asarray(lower)) & (p <= jnp.asarray(upper)))
        & jnp.all(correlation_minors(backbone) >= margin)
    )


def _coordinates(template, names, bounds, scales, margin):
    """The scales and the bounds in the coordinates x, for a template inside the bounds."""
    s = jnp.asarray(scales, dtype=jnp.float64)
    start = jnp.asarray([getattr(template, name) for name in names])
    low, high = ((jnp.asarray(b) - start) / s for b in bounds)
    if not bool(jnp.all((low < 0) & (high > 0))) or not _admissible(
        template, names, *bounds, margin
    ):
        raise ValueError("the template must lie inside the bounds and the correlation margin")
    return s, low, high


def fit_bergomi(
    template,
    times,
    increments,
    maturities,
    log_strikes,
    is_call,
    lower,
    upper,
    weights,
    vegas,
    radii=(0.25,) * 8,
    cumulative_radius=1.5,
    leverage=None,
    names=PARAMETERS,
    bounds=(LOWER, UPPER),
    scales=SCALES,
    margin=1e-4,
    solver=SOLVER,
    implied=False,
):
    """The backbones after each step, starting from the template.

    increments are the replications of Brownian increments (paths, steps, 3) on the grid times,
    lower and upper the quote intervals (equal for midpoints), vegas the scales of the loss and
    bounds the lower and upper bounds of the parameters names. With implied, the intervals and
    scales are in implied volatility.
    """
    s, low, high = _coordinates(template, names, bounds, scales, margin)
    x, fitted = jnp.zeros(s.size), []
    for radius in radii:
        prices, jacobian = bergomi_prices(
            template, x, times, increments, maturities, log_strikes, is_call, leverage, names, s
        )
        if implied:
            prices, jacobian = volatility_coordinates(
                prices, jacobian, maturities, log_strikes, is_call
            )
        linear = linearize(prices, jacobian, lower, upper, weights, vegas)
        loss = local_loss(linear, jnp.eye(x.size), radius, solver)
        constraints = Constraints(
            template, s, low, high, margin, x, loss.transform, radius, cumulative_radius, names
        )
        x = x + slsqp(
            loss,
            lambda z, c=constraints: margins(z, c),
            lambda z, c=constraints: margin_jacobian(z, c),
            solver,
        )
        backbone = displaced(template, x, s, names)
        if not _admissible(backbone, names, *bounds, margin):
            raise ConvergenceError("the step leaves the bounds or the correlation margin")
        fitted.append(backbone)
    return fitted


_loss_gradient = jax.jit(jax.grad(linearized_loss))


def fit_bergomi_interior_point(
    template,
    times,
    increments,
    maturities,
    log_strikes,
    is_call,
    lower,
    upper,
    weights,
    vegas,
    leverage=None,
    names=PARAMETERS,
    bounds=(LOWER, UPPER),
    scales=SCALES,
    margin=1e-4,
    options=INTERIOR_POINT,
    implied=False,
):
    """The backbone minimizing the loss under the bounds and the correlation constraints, by
    the interior point method (scipy's trust-constr with the options); arguments as for
    fit_bergomi."""
    s, low, high = _coordinates(template, names, bounds, scales, margin)
    # Minors that do not involve a fitted correlation are constant and left out.
    varying = [i for i, name in enumerate(("rho1", "rho2", "rho_z")) if name in names]
    varying = np.asarray(varying + [3] if varying else [], dtype=int)

    def loss(x):
        prices, jacobian = bergomi_prices(
            template,
            jnp.asarray(x),
            times,
            increments,
            maturities,
            log_strikes,
            is_call,
            leverage,
            names,
            s,
        )
        if implied:
            prices, jacobian = volatility_coordinates(
                prices, jacobian, maturities, log_strikes, is_call
            )
        linear = linearize(prices, jacobian, lower, upper, weights, vegas)
        zero = jnp.zeros(s.size)
        return float(linearized_loss(zero, linear)), np.asarray(_loss_gradient(zero, linear))

    def minors(x):
        return correlation_minors(displaced(template, x, s, names))[varying]

    constraints = []
    if varying.size:
        constraints.append(
            NonlinearConstraint(
                lambda x: np.asarray(minors(jnp.asarray(x))),
                margin,
                np.inf,
                jac=lambda x: np.asarray(jax.jacfwd(minors)(jnp.asarray(x))),
                hess=BFGS(),
                keep_feasible=True,
            )
        )
    result = minimize(
        loss,
        np.zeros(s.size),
        method="trust-constr",
        jac=True,
        hess=BFGS(),
        bounds=Bounds(np.asarray(low), np.asarray(high), keep_feasible=True),
        constraints=constraints,
        options=dict(options, verbose=0),
    )
    backbone = displaced(template, jnp.asarray(result.x), s, names)
    if not result.success or not _admissible(backbone, names, *bounds, margin):
        raise ConvergenceError(f"trust-constr: {result.message}")
    return backbone
