"""The progressive calibration: one regression per time step, in a single forward pass.

On the grid 0 = t_0 < t_1 < ... < t_n all paths start at X_{t_0} = 0. At every node t_j,
j = 0, ..., n - 1, the calibration

1. fits the conditional mean m_j(z) of Y_{t_j} given the coordinates z = z(t_j, X_{t_j}), by a
   regression of Y_{t_j} on z(t_j, X_{t_j}) without offset;
2. sets the leverage factor L(t_j, x) = sigma_D(t_j, x) / sqrt(max(m_j(z(t_j, x)), f)), with an
   optional floor f of the conditional mean;
3. advances the paths to t_{j+1} by one Euler step with L(t_j, .).

The conditional mean is piecewise constant in time, m(t, z) = m_j(z) for t_j <= t < t_{j+1}, and
L(t, x) = sigma_D(t, x) / sqrt(max(m(t, z(t, x)), f)); the Euler scheme reads it at the nodes
only, so the paths must be simulated on the same grid. Every node is fitted once; there is no
iteration. The method is the particle method of Guyon and Henry-Labordere (Risk, 2012) with a
regression in place of the kernel estimate; with quantile bins as regression it is the method of
van der Stoep, Grzelak and Oosterlee (Int. J. Theor. Appl. Finance 17, 2014).
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.functions import compiled
from locsvol.regression.means import fitted_mean, per_time
from locsvol.simulation.euler import euler_step


def of_mean(sigma, mean, floor=None):
    """sigma / sqrt(max(m, f)) where sigma and the mean m are finite and positive, else nan."""
    valid = jnp.isfinite(sigma) & (sigma > 0) & jnp.isfinite(mean) & (mean > 0)
    selected = mean if floor is None else jnp.maximum(mean, floor)
    value = sigma / jnp.sqrt(jnp.where(valid, selected, jnp.nan))
    return jnp.where(valid & jnp.isfinite(value) & (value > 0), value, jnp.nan)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["target", "chart", "means"],
    meta_fields=["floor"],
)
@dataclass(frozen=True)
class Progressive:
    """L(t, x) = sigma_D(t, x) / sqrt(max(m(t, z(t, x)), f)) for a mean m that is piecewise
    constant in time (regression.means.PerTime); nan outside its grid."""

    target: object
    chart: object
    means: object
    floor: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "target", compiled(self.target))
        object.__setattr__(self, "chart", compiled(self.chart))

    def __call__(self, t, x):
        t, x = jnp.asarray(t), jnp.asarray(x)
        value = of_mean(self.target(t, x), self.means(t, self.chart(t, x)), self.floor)
        t, x = jnp.broadcast_arrays(t, x)
        return jnp.where(jnp.isfinite(t) & (t >= 0) & jnp.isfinite(x), value, jnp.nan)


_step = jax.jit(euler_step)


def progressive(target, chart, paths, regression, floor=None, weights=None):
    """The leverage factor of the progressive calibration on the grid of paths.

    target is the local volatility sigma_D(t, x), chart the state coordinates z(t, x) of shape
    (..., d), paths the simulated variance factor with its spot increments, and regression a
    regression method (GammaRegression, GammaSplines, PositiveMeanNetwork, QuantileBins,
    CustomRegression), or a sequence of them, one per time step, for instance networks with
    distinct streams. weights (paths, nodes) weigh the observations, equally by default.
    """
    times, variance = paths.times, paths.variance
    steps = times.size - 1
    regressions = list(regression) if isinstance(regression, list | tuple) else [regression] * steps
    if len(regressions) != steps:
        raise ValueError("one regression per time step is required")
    weights = jnp.ones_like(variance) if weights is None else jnp.asarray(weights)
    x = jnp.zeros(variance.shape[0])
    means = []
    for j in range(steps):
        mean = fitted_mean(regressions[j], chart(times[j], x), variance[:, j], weights[:, j])
        means.append(mean)
        current = Progressive(target, chart, per_time(times[j : j + 2], [mean]), floor)
        dt = times[j + 1] - times[j]
        x = _step(current, times[j], dt, x, variance[:, j], paths.spot_increments[:, j])
        if not bool(jnp.all(jnp.isfinite(x))):
            raise ValueError(f"no finite leverage factor on all paths at time step {j}")
    return Progressive(target, chart, per_time(times, means), floor)
