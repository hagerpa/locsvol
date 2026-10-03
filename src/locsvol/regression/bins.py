"""Conditional means by quantile bins.

The N observations (x_n, y_n) of one coordinate, sorted by x, are split into
B = max(1, min(bin_count, N // minimum_bin_size)) bins of consecutive observations of equal count,
the first N mod B bins one larger. Each bin gives a node (xbar_k, ybar_k), the means of its
coordinates and responses; nodes with equal xbar are merged, weighted by their counts. The log
means log ybar_k are interpolated by the monotone piecewise cubic of Fritsch and Carlson (SIAM J.
Numer. Anal. 17, 1980), as SciPy's PchipInterpolator, and the mean is constant beyond the outer
nodes ('mean_log_pchip').

The bins of van der Stoep, Grzelak and Oosterlee (The Heston stochastic-local volatility model:
efficient Monte Carlo simulation, Int. J. Theor. Appl. Finance 17, 2014) instead read the
coordinate as a log level, x = log S: bin k reaches from the last observation of bin k - 1 (the
first observation, for k = 0) to its own last one, its node is the midpoint of these two levels
S, and the nodes are joined by straight lines in the level S ('midpoint_level_linear').

All observations carry the same weight. The fitted table is padded to min(bin_count, N) nodes,
so that the tables of fits with equally many observations stack.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.regression.cubic import cubic

REPRESENTATIONS = ("mean_log_pchip", "midpoint_level_linear")


@partial(jax.jit, static_argnames="count")
def bin_statistics(x, y, count):
    """Mean coordinates, mean responses and sizes of count bins of the sorted observations."""
    n = x.size
    sizes = n // count + (jnp.arange(count) < n % count)
    labels = jnp.searchsorted(jnp.cumsum(sizes), jnp.arange(n), side="right")
    centers = jax.ops.segment_sum(x, labels, num_segments=count) / sizes
    means = jax.ops.segment_sum(y, labels, num_segments=count) / sizes
    return centers, means, sizes


@jax.jit
def pchip_coefficients(knots, values):
    """Coefficients (G - 1, 4), in descending powers of x - x_i, of the monotone cubic of
    Fritsch and Carlson through (knots, values): weighted harmonic means of the secants as inner
    slopes, one-sided three-point slopes at the ends, limited to keep their sign and size."""
    h = jnp.diff(knots)
    secants = jnp.diff(values) / h
    if knots.size == 2:
        slopes = jnp.repeat(secants, 2)
    else:
        left, right = secants[:-1], secants[1:]
        same = (left != 0) & (right != 0) & (jnp.sign(left) == jnp.sign(right))
        w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
        harmonic = (w1 + w2) / (w1 / jnp.where(same, left, 1.0) + w2 / jnp.where(same, right, 1.0))

        def end(h0, h1, m0, m1):
            slope = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
            return jnp.where(
                jnp.sign(slope) != jnp.sign(m0),
                0.0,
                jnp.where(
                    (jnp.sign(m0) != jnp.sign(m1)) & (jnp.abs(slope) > 3 * jnp.abs(m0)),
                    3 * m0,
                    slope,
                ),
            )

        slopes = jnp.concatenate(
            (
                end(h[0], h[1], secants[0], secants[1])[None],
                jnp.where(same, harmonic, 0.0),
                end(h[-1], h[-2], secants[-1], secants[-2])[None],
            )
        )
    return jnp.stack(
        (
            (slopes[:-1] + slopes[1:] - 2 * secants) / h**2,
            (3 * secants - 2 * slopes[:-1] - slopes[1:]) / h,
            slopes[:-1],
            values[:-1],
        ),
        axis=-1,
    )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["grid", "values", "coefficients", "count"],
    meta_fields=["representation"],
)
@dataclass(frozen=True)
class Bins:
    """The fitted mean of quantile bins: the first count nodes of grid with their means."""

    grid: jax.Array
    values: jax.Array
    coefficients: jax.Array
    count: jax.Array
    representation: str = "mean_log_pchip"

    def __call__(self, z):
        """The mean at coordinates z of shape (..., 1), shape (...)."""
        x = jnp.asarray(z, dtype=jnp.float64)[..., 0]

        def interpolate():
            bounded = jnp.clip(x, self.grid[0], self.grid[self.count - 1])
            index = jnp.clip(
                jnp.searchsorted(self.grid, bounded, side="right") - 1, 0, self.count - 2
            )
            if self.representation == "midpoint_level_linear":
                dx = jnp.exp(bounded) - jnp.exp(self.grid[index])
            else:
                dx = bounded - self.grid[index]
            value = cubic(self.coefficients[index], dx)
            return jnp.exp(value) if self.representation == "mean_log_pchip" else value

        mean = jax.lax.cond(self.count == 1, lambda: jnp.full_like(x, self.values[0]), interpolate)
        return jnp.where(jnp.isfinite(x) & jnp.isfinite(mean) & (mean > 0), mean, jnp.nan)


@dataclass(frozen=True)
class QuantileBins:
    """The quantile-bin regression of the module docstring."""

    bin_count: int
    minimum_bin_size: int = 1
    representation: str = "mean_log_pchip"

    def __post_init__(self):
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"the representation must be one of {REPRESENTATIONS}")

    def fit(self, x, y, weights):
        """The fitted mean from coordinates x of shape (n, 1), responses y and equal weights."""
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)
        w = jnp.asarray(weights)
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("quantile bins need one coordinate, of shape (n, 1)")
        if not bool(jnp.all(w > 0) & jnp.all(w == w[0])):
            raise ValueError("quantile bins fit observations of equal positive weight")
        order = jnp.argsort(x[:, 0], stable=True)
        x, y = x[order, 0], y[order]
        count = max(1, min(self.bin_count, y.size // self.minimum_bin_size))
        centers, means, sizes = bin_statistics(x, y, count)
        if self.representation == "midpoint_level_linear":
            last = jnp.cumsum(sizes) - 1
            first = jnp.concatenate((jnp.zeros(1, last.dtype), last[:-1]))
            centers = jnp.log(0.5 * (jnp.exp(x[first]) + jnp.exp(x[last])))
        grid, inverse = jnp.unique(centers, return_inverse=True)
        counts = jax.ops.segment_sum(sizes, inverse, num_segments=grid.size)
        values = jax.ops.segment_sum(means * sizes, inverse, num_segments=grid.size) / counts
        if grid.size == 1:
            coefficients = jnp.zeros((0, 4))
        elif self.representation == "mean_log_pchip":
            coefficients = pchip_coefficients(grid, jnp.log(values))
        else:
            slopes = jnp.diff(values) / jnp.diff(jnp.exp(grid))
            zeros = jnp.zeros_like(slopes)
            coefficients = jnp.stack((zeros, zeros, slopes, values[:-1]), axis=-1)
        if not bool(
            jnp.all(jnp.isfinite(values) & (values > 0)) & jnp.all(jnp.isfinite(coefficients))
        ):
            raise ValueError("the bins give means that are not finite and positive")
        capacity = min(self.bin_count, y.size)
        return Bins(
            jnp.pad(grid, (0, capacity - grid.size), mode="edge"),
            jnp.pad(values, (0, capacity - grid.size), mode="edge"),
            jnp.pad(coefficients, ((0, max(capacity - 1, 1) - coefficients.shape[0]), (0, 0))),
            jnp.asarray(grid.size, dtype=jnp.int32),
            self.representation,
        )
