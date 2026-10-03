"""The fixed-point iteration with coefficients interpolated in time between knots.

At knots 0 = T_0 < T_1 < ... < T_K = T of the time grid, the conditional mean E[Y_t | X_t = x] is
approximated by exp(b(z(T_k, x))^T gamma_k), with a basis b of the state coordinate of a chart
z(t, x), for instance the Chebyshev polynomials T_j(u) of the SSVI chart, ChebyshevTensor((0, q)).
Between the knots the coefficients are interpolated linearly in time,

    gamma(t) = (1 - u) gamma_k + u gamma_{k+1},   u = (t - T_k) / (T_{k+1} - T_k),

which gives the leverage factor

    L(t, x) = sigma_D(t, x) / sqrt(max(exp(b(z(t, x))^T gamma(t)), f)),

with an optional floor f of the conditional mean. An update fits, at a knot T_k, the regression
of Y_{T_k} on z(T_k, X_{T_k}) for paths simulated with the current leverage factor:

- direct: the regression without offset gives coefficients c_k, the proposal gamma_k' = c_k;
- residual: the regression with the offset log(sigma_D^2 / L^2) = b^T gamma_k gives a correction
  d_k, the proposal gamma_k' = gamma_k + d_k.

With a relaxation a in (0, 1], gamma_k becomes (1 - a) gamma_k + a gamma_k', which is the geometric
update L^{1-a} L'^a of the leverage factor when there is no floor. A floor admits only the direct
update with a = 1.

The iteration runs on one of two schedules:

- full horizon: every iteration simulates the paths on [0, T] with the current leverage factor
  and updates all knots, the paths starting at X = 0 at T_0;
- interval by interval ('interval_progressive'): the knot T_0, where all paths sit at X = 0, is
  fitted once; then for k = 0, ..., K - 1 every iteration simulates the paths on [T_k, T_{k+1}]
  from their values at T_k and updates gamma_{k+1} alone, and the paths are finally advanced to
  T_{k+1} with the last leverage factor.
"""

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from locsvol.calibration.progressive import of_mean
from locsvol.functions import compiled
from locsvol.simulation.euler import log_spot


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["target", "chart", "knots", "coefficients"],
    meta_fields=["basis", "floor"],
)
@dataclass(frozen=True)
class TimeInterpolated:
    """L(t, x) = sigma_D(t, x) / sqrt(max(exp(b(z(t, x))^T gamma(t)), f)) with the rows
    gamma_k of coefficients (K + 1, B) at the knots; see the module docstring."""

    target: object
    chart: object
    basis: object
    knots: jax.Array
    coefficients: jax.Array
    floor: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "target", compiled(self.target))
        object.__setattr__(self, "chart", compiled(self.chart))

    def coefficients_at(self, t):
        """gamma(t), of shape t.shape + (B,); nan outside [T_0, T_K]."""
        t = jnp.asarray(t)
        j = jnp.clip(jnp.searchsorted(self.knots, t, side="right") - 1, 0, self.knots.size - 2)
        u = (t - self.knots[j]) / (self.knots[j + 1] - self.knots[j])
        value = (1 - u[..., None]) * self.coefficients[j] + u[..., None] * self.coefficients[j + 1]
        valid = jnp.isfinite(t) & (t >= self.knots[0]) & (t <= self.knots[-1])
        return jnp.where(valid[..., None], value, jnp.nan)

    def log_mean(self, t, x):
        """b(z(t, x))^T gamma(t)."""
        t, x = jnp.asarray(t), jnp.asarray(x)
        features = self.basis(self.chart(t, x))
        gamma = self.coefficients_at(t)
        if t.ndim == 0:
            return features @ gamma
        return jnp.sum(features * jnp.broadcast_to(gamma, features.shape), axis=-1)

    def __call__(self, t, x):
        t, x = jnp.asarray(t), jnp.asarray(x)
        value = of_mean(self.target(t, x), jnp.exp(self.log_mean(t, x)), self.floor)
        t, x = jnp.broadcast_arrays(t, x)
        return jnp.where(jnp.isfinite(t) & (t >= 0) & jnp.isfinite(x), value, jnp.nan)


def knot_indices(times, knots):
    """Indices of the knots among the nodes of the time grid, which must contain them all and
    share its first and last node."""
    times, knots = np.asarray(times), np.asarray(knots)
    indices = np.searchsorted(times, knots)
    if (
        knots.ndim != 1
        or knots.size < 2
        or np.any(np.diff(knots) <= 0)
        or np.any(indices >= times.size)
        or not np.array_equal(times[np.minimum(indices, times.size - 1)], knots)
        or knots[0] != times[0]
        or knots[-1] != times[-1]
    ):
        raise ValueError("the knots must be increasing nodes of the grid, including both ends")
    return indices


@jax.jit
def _advance(leverage, times, variance, increments, x):
    """X at the last node of times from X = x at the first, by the Euler scheme."""
    return log_spot(leverage, times, variance, increments, start=x)[:, -1]


def _proposal(leverage, regression, k, x, y, weights, formulation):
    """gamma_k' from the regression at knot T_k of the responses y on the paths X_{T_k} = x."""
    t = leverage.knots[k]
    z = leverage.chart(t, x)
    if formulation == "direct":
        return regression.fit(z, y, weights)
    offset = 2 * jnp.log(leverage.target(t, x)) - 2 * jnp.log(leverage(t, x))
    return leverage.coefficients[k] + regression.fit(z, y, weights, offset)


def _relaxed(current, proposed, relaxation):
    """(1 - a) gamma + a gamma', keeping rows that do not change."""
    if relaxation == 1:
        return proposed
    blended = (1 - relaxation) * current + relaxation * proposed
    return jnp.where(current == proposed, current, blended)


def _check(leverage, regression, relaxation, formulation):
    if formulation not in ("direct", "residual"):
        raise ValueError("the formulation must be 'direct' or 'residual'")
    if not 0 < relaxation <= 1:
        raise ValueError("the relaxation must lie in (0, 1]")
    if leverage.floor is not None and (formulation == "residual" or relaxation != 1):
        raise ValueError("a floor admits only the direct update with relaxation one")
    if regression.basis != leverage.basis:
        raise ValueError("the regression must fit the basis of the leverage factor")


def _weights(paths, weights):
    return jnp.ones_like(paths.variance) if weights is None else jnp.asarray(weights)


def full_horizon(
    leverage, paths, regression, iterations, relaxation=1.0, formulation="direct", weights=None
):
    """The leverage factors of the iterations 1, ..., iterations on the full horizon.

    leverage is the initial TimeInterpolated factor, paths the simulated variance factor, whose
    grid contains the knots, regression for instance a GammaRegression on the leverage factor's
    basis, and weights (paths, nodes) the weights of the observations, equal by default.
    """
    _check(leverage, regression, relaxation, formulation)
    indices = knot_indices(paths.times, leverage.knots)
    weights = _weights(paths, weights)
    iterates = []
    for _ in range(iterations):
        x = paths.log_spot(leverage)[:, indices]
        proposed = jnp.stack(
            [
                _proposal(
                    leverage,
                    regression,
                    k,
                    x[:, k],
                    paths.variance[:, i],
                    weights[:, i],
                    formulation,
                )
                for k, i in enumerate(indices)
            ]
        )
        coefficients = _relaxed(leverage.coefficients, proposed, relaxation)
        leverage = replace(leverage, coefficients=coefficients)
        iterates.append(leverage)
    return iterates


def interval_progressive(
    leverage, paths, regression, iterations, relaxation=1.0, formulation="direct", weights=None
):
    """The leverage factor after the iterations on every interval [T_k, T_{k+1}].

    iterations is the number of iterations per interval, one number or one per interval; the
    other arguments are those of full_horizon.
    """
    _check(leverage, regression, relaxation, formulation)
    indices = knot_indices(paths.times, leverage.knots)
    weights = _weights(paths, weights)
    counts = np.broadcast_to(iterations, (indices.size - 1,))
    x = jnp.zeros(paths.variance.shape[0])

    def update(leverage, k, x, a):
        i = indices[k]
        proposed = _proposal(
            leverage, regression, k, x, paths.variance[:, i], weights[:, i], formulation
        )
        row = _relaxed(leverage.coefficients[k], proposed, a)
        return replace(leverage, coefficients=leverage.coefficients.at[k].set(row))

    leverage = update(leverage, 0, x, 1.0)
    for k, count in enumerate(counts):
        start, stop = indices[k], indices[k + 1]
        segment = (
            paths.times[start : stop + 1],
            paths.variance[:, start:stop],
            paths.spot_increments[:, start:stop],
        )
        for _ in range(int(count)):
            leverage = update(leverage, k + 1, _advance(leverage, *segment, x), relaxation)
        x = _advance(leverage, *segment, x)
        if not bool(jnp.all(jnp.isfinite(x))):
            raise ValueError(f"paths without a finite value at knot {k + 1}")
    return leverage
