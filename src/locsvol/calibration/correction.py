"""Correction steps of the leverage factor against bid and ask quotes (Section 3.3 of the paper).

The parametrized leverage factor L_beta = L_0 exp(b^T beta) has relatively few coefficients, so
it is calibrated directly to the quotes by Monte Carlo, starting from an iterate beta_0 of the
fixed-point iteration. The loss is the vega-scaled distance of the model prices to the bid-ask
intervals (see gauss_newton): for a liquid quote its bid-ask interval, for every other quote the
smallest interval containing its bid-ask interval and its price at beta_0, so that such a quote
may improve but costs only by moving further out.

Each correction step is a regularized Gauss-Newton step in the coefficients (gauss_newton). Its
metric is the Gram matrix

    M(beta) = (1/m) sum_k sum_j w_j b(t_j, X^k_j) b(t_j, X^k_j)^T

over the m paths X^k of the model with L_beta and the trapezoidal time weights w_j, so that
delta^T M(beta) delta is the mean-square change of log L along the paths. For the Chebyshev
tensor basis b(t, x) = T(tau(t)) (x) T(u(t, x)) of the chart coordinates, M follows from the
moments sum_j (w_j / m) T(tau_j) (x) sum_k T(u(t_j, X^k_j)) (regression.chebyshev). The step
delta from beta = beta_0 + theta satisfies

    delta^T M(beta) delta <= rho_i^2                                (trust region of step i),
    (theta + delta)^T M(beta_0) (theta + delta) <= R^2              (cumulative change),
    |log L_{beta + delta}(t_n, x_n) - log L_{beta_0}(t_n, x_n)| <= b  (support points),

so the corrected leverage factor stays within a factor e^{+-b} of the starting point on the
support. Since log L is linear in beta, the last bounds are linear in the coefficients. The
support points lie at time nodes thinned uniformly in index, at states equally spaced in the
chart's state coordinate across its envelope, beyond which L_beta is constant in the state.
Without these constraints, the step may instead minimize the penalized local loss by the Newton
iteration of gauss_newton.

Prices come from independent sets of paths (replications), their derivatives in the
coefficients by pathwise differentiation (pricing.monte_carlo), and M from the same paths.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from locsvol.calibration.gauss_newton import (
    LocalSolver,
    linearize,
    local_loss,
    newton,
    slsqp,
    volatility_coordinates,
)
from locsvol.errors import ConvergenceError
from locsvol.leverage.chart import Chart
from locsvol.leverage.parametrized import Parametrized
from locsvol.pricing.monte_carlo import (
    chunks,
    log_spot_derivatives,
    maturity_groups,
    maturity_nodes,
    mean_payoffs_and_derivatives,
)
from locsvol.regression.chebyshev import chebyshev

SOLVER = LocalSolver()


def widened_intervals(bid, ask, liquid, prices):
    """[bid, ask] for liquid quotes, otherwise the smallest interval containing [bid, ask] and
    the price."""
    lower = jnp.where(liquid, bid, jnp.minimum(bid, prices))
    upper = jnp.where(liquid, ask, jnp.maximum(ask, prices))
    return lower, upper


_log_moneyness = jax.jit(Chart.log_moneyness)


def support_points(chart, times, nodes, states):
    """Support points (t_n, x_n), sorted and without repetitions: at nodes time nodes thinned
    uniformly in index, first and last kept, states equally spaced in the state coordinate
    between the bounds of the chart's envelope."""
    if chart.envelope is None:
        raise ValueError("the support spans the envelope of the chart")
    grid = np.asarray(times)
    t = jnp.asarray(grid[np.unique(np.round(np.linspace(0, grid.size - 1, nodes)).astype(int))])
    envelope_times, lower, upper = chart.envelope
    lower, upper = jnp.interp(t, envelope_times, lower), jnp.interp(t, envelope_times, upper)
    u = lower[:, None] + (upper - lower)[:, None] * jnp.linspace(0.0, 1.0, states)
    maturities = jnp.broadcast_to(t[:, None], u.shape)
    x = _log_moneyness(chart, maturities, u)
    points = np.unique(np.stack((np.ravel(maturities), np.ravel(x)), axis=1), axis=0)
    return jnp.asarray(points[:, 0]), jnp.asarray(points[:, 1])


@jax.jit
def path_gram(leverage, times, x, weights):
    """M(beta) of the module docstring for paths x (paths, nodes) of the model with L_beta."""
    p, q = leverage.basis.degrees
    clock = leverage.chart.time(times)

    def node(moments, j):
        state = chebyshev(leverage.chart(times[j], x[:, j])[:, 1], 2 * q)
        moment = jnp.outer(chebyshev(clock[j], 2 * p), jnp.sum(state, axis=0))
        return moments + moment * (weights[j] / x.shape[0]), None

    moments, _ = jax.lax.scan(node, jnp.zeros((2 * p + 1, 2 * q + 1)), jnp.arange(times.size))
    return leverage.basis.gram(moments)


def sensitivities(leverage, paths, maturities, log_strikes, is_call, time_weights):
    """Prices (quotes,), their derivatives in the coefficients (quotes, size) and M(beta), as
    means over the chunks of paths of all replications."""
    unique, groups = maturity_groups(maturity_nodes(paths[0].times, maturities))
    k, call = jnp.asarray(log_strikes), jnp.asarray(is_call)
    size = leverage.coefficients.size
    prices, jacobian, gram = jnp.zeros(k.size), jnp.zeros((k.size, size)), jnp.zeros((size, size))
    for replication in paths:
        count = replication.variance.shape[0]
        for part in chunks(count):
            weight = (part.stop - part.start) / count / len(paths)
            x, d = log_spot_derivatives(
                leverage,
                replication.times,
                replication.variance[part],
                replication.spot_increments[part],
                jnp.asarray(unique),
            )
            for slot, rows in enumerate(groups):
                value, derivative = mean_payoffs_and_derivatives(
                    x[:, unique[slot]], d[slot], k[rows], call[rows]
                )
                prices = prices.at[rows].add(weight * value)
                jacobian = jacobian.at[rows].add(weight * derivative)
            gram = gram + weight * path_gram(leverage, replication.times, x, time_weights)
    return prices, jacobian, gram


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "leverage",
        "displacement",
        "transform",
        "metric",
        "initial_metric",
        "radius",
        "cumulative_radius",
        "times",
        "states",
        "initial_log",
        "bound",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class Constraints:
    """The constraints of one step on z, with delta = transform z and theta = displacement:
    leverage is L_{beta_0}, initial_log is log L_{beta_0} at the support points (times, states),
    and a constraint is absent where its radius or bound is None."""

    leverage: Parametrized
    displacement: jax.Array
    transform: jax.Array
    metric: jax.Array
    initial_metric: jax.Array
    radius: float | None
    cumulative_radius: float | None
    times: jax.Array
    states: jax.Array
    initial_log: jax.Array
    bound: float | None


@jax.jit
def margins(z, constraints):
    """Margins of the constraints, nonnegative where they hold: trust region, cumulative change,
    then the upper and the lower bounds at the support points."""
    c = constraints
    delta = c.transform @ z
    theta = c.displacement + delta
    parts = []
    if c.radius is not None:
        parts.append(jnp.atleast_1d(1 - delta @ c.metric @ delta / c.radius**2))
    if c.cumulative_radius is not None:
        parts.append(jnp.atleast_1d(1 - theta @ c.initial_metric @ theta / c.cumulative_radius**2))
    if c.bound is not None:
        movement = jnp.log(c.leverage.shifted(theta)(c.times, c.states)) - c.initial_log
        parts.extend((1 - movement / c.bound, 1 + movement / c.bound))
    return jnp.concatenate(parts)


margin_jacobian = jax.jit(jax.jacfwd(margins))


def correction_steps(
    leverage,
    paths,
    maturities,
    log_strikes,
    is_call,
    lower,
    upper,
    weights,
    vegas,
    time_weights,
    radii,
    cumulative_radius=None,
    bound=None,
    support=None,
    solver=SOLVER,
    implied=False,
):
    """The leverage factors after each correction step, starting from the parametrized
    leverage factor L_{beta_0}.

    paths are independent replications of the variance factor with their spot increments,
    lower and upper the quote intervals (see widened_intervals), vegas the scales of the
    loss, time_weights the w_j of M on the time grid of the paths, radii the trust region
    radii rho_i of the steps, and support the points (t_n, x_n) of the bound b. A radius,
    cumulative radius or bound of None imposes no such constraint; the Newton solver
    requires that none is imposed. With implied, the intervals and scales are in implied
    volatility. A step whose local problem the solver does not solve ends the corrections; the
    leverage factors of the steps before it are returned.
    """
    constrained = cumulative_radius is not None or bound is not None
    if solver.method == "newton" and (constrained or any(r is not None for r in radii)):
        raise ValueError("the Newton solver takes no trust region, cumulative radius or bound")
    if bound is not None and support is None:
        raise ValueError("a bound needs support points")
    if any(np.shape(time_weights) != replication.times.shape for replication in paths):
        raise ValueError("the time weights must have one entry per node of the paths")
    times, states = support if support is not None else (jnp.zeros(0), jnp.zeros(0))
    initial_log = jnp.log(leverage(times, states))
    theta = jnp.zeros_like(leverage.coefficients)
    initial_metric, corrected = None, []
    for radius in radii:
        current = leverage.shifted(theta)
        prices, jacobian, metric = sensitivities(
            current, paths, maturities, log_strikes, is_call, time_weights
        )
        if not bool(jnp.all(jnp.isfinite(metric))):
            raise ValueError("the Gram matrix along the paths is not finite")
        initial_metric = metric if initial_metric is None else initial_metric
        if implied:
            prices, jacobian = volatility_coordinates(
                prices, jacobian, maturities, log_strikes, is_call
            )
        linear = linearize(prices, jacobian, lower, upper, weights, vegas)
        loss = local_loss(linear, metric, 1.0 if radius is None else radius, solver)
        try:
            if solver.method == "newton":
                delta = newton(loss, solver)
            elif radius is None and not constrained:
                delta = slsqp(loss, None, None, solver)
            else:
                constraints = Constraints(
                    leverage,
                    theta,
                    loss.transform,
                    metric,
                    initial_metric,
                    radius,
                    cumulative_radius,
                    times,
                    states,
                    initial_log,
                    bound,
                )
                delta = slsqp(
                    loss,
                    lambda z, c=constraints: margins(z, c),
                    lambda z, c=constraints: margin_jacobian(z, c),
                    solver,
                )
        except ConvergenceError:
            break
        theta = theta + delta
        corrected.append(leverage.shifted(theta))
    return corrected
