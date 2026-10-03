"""A positive function of (t, x) tabulated on a rectangle in (t, u), and the capped base.

The values are given at times t_i and at equally spaced nodes u_k of the state coordinate u of a
chart, and interpolated bilinearly in (t, u). Since the nodes are equally spaced in u, they are
dense in x near the money and sparse in the wings. A query outside the rectangle takes the value
at its edge.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from locsvol.functions import compiled
from locsvol.leverage.chart import Chart


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["times", "nodes", "values", "chart"],
    meta_fields=[],
)
@dataclass(frozen=True)
class Table:
    """Values (times, nodes) of a positive function on the chart's state coordinate."""

    times: jax.Array
    nodes: jax.Array
    values: jax.Array
    chart: Chart

    def __call__(self, t, x):
        t, x = jnp.asarray(t), jnp.asarray(x)
        span = (self.nodes[-1] - self.nodes[0]) / (self.nodes.size - 1)
        positions = jnp.arange(self.times.size, dtype=jnp.float64)
        if t.ndim == 0:
            u = self.chart(t, x)[..., -1]
            position = jnp.interp(t, self.times, positions)
            low = jnp.clip(jnp.floor(position), 0, self.times.size - 2)
            weight = position - low
            index = low.astype(jnp.int32)
            row = (1.0 - weight) * self.values[index] + weight * self.values[index + 1]
            columns = (jnp.clip(u, self.nodes[0], self.nodes[-1]) - self.nodes[0]) / span
            value = map_coordinates(row, [columns], order=1, mode="nearest")
        else:
            t, x = jnp.broadcast_arrays(t, x)
            u = self.chart(t, x)[..., -1]
            rows = jnp.interp(t, self.times, positions)
            columns = (jnp.clip(u, self.nodes[0], self.nodes[-1]) - self.nodes[0]) / span
            value = map_coordinates(self.values, [rows, columns], order=1, mode="nearest")
        t = jnp.broadcast_to(t, jnp.shape(value))
        valid = jnp.isfinite(t) & (t >= 0) & jnp.isfinite(x) & jnp.isfinite(value) & (value > 0)
        return jnp.where(valid, value, jnp.nan)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["function"],
    meta_fields=["cap", "lower", "upper"],
)
@dataclass(frozen=True)
class Capped:
    """min(sigma(t, clip(x, lower, upper)), cap): a local volatility read inside [lower, upper]
    and capped, for instance the base L_0 = min(sigma_Dup, 5) of the paper."""

    function: object
    cap: float | None
    lower: float
    upper: float

    def __post_init__(self):
        object.__setattr__(self, "function", compiled(self.function))

    def __call__(self, t, x):
        value = self.function(t, jnp.clip(jnp.asarray(x), self.lower, self.upper))
        return value if self.cap is None else jnp.minimum(value, self.cap)


def table_times(
    expiries, horizon, head_start, head_end, head_count, tail_step, offsets, required=()
):
    """Union of a geometric head, a uniform tail, every expiry shifted by each offset, the
    horizon and the required times, within [0, horizon]."""
    head = np.geomspace(head_start, head_end, head_count)
    tail = np.arange(head_end, horizon, tail_step)
    marks = np.asarray(expiries)[:, None] + np.asarray(offsets)[None, :]
    times = np.concatenate([head, tail, marks.ravel(), [horizon], np.asarray(required).ravel()])
    return jnp.asarray(np.unique(times[(times >= 0.0) & (times <= horizon)]))


def tabulate(function, chart, times, state_limit=0.995, state_count=801):
    """The table of a positive function at the times and at state_count equally spaced nodes of
    the state coordinate in [-state_limit, state_limit]. Where the function has no positive
    finite value, a node takes the value of its nearest neighbour toward the money (u = 0) in the
    same row, so that the table is constant beyond the last valid node."""
    nodes = jnp.linspace(-state_limit, state_limit, state_count)
    grid = jnp.broadcast_to(times[:, None], (times.size, nodes.size))
    x = chart.log_moneyness(grid, jnp.broadcast_to(nodes[None, :], grid.shape))
    values = np.array(function(grid, x))
    centre = int(np.argmin(np.abs(np.asarray(nodes))))
    for columns in (range(centre + 1, state_count), range(centre - 1, -1, -1)):
        for j in columns:
            inner = j - 1 if j > centre else j + 1
            invalid = ~(np.isfinite(values[:, j]) & (values[:, j] > 0))
            values[invalid, j] = values[invalid, inner]
    return Table(times, nodes, jnp.asarray(values), chart)
