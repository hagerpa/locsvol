"""The particle method of Guyon and Henry-Labordere (Being particular about calibration, Risk,
2012; Nonlinear Option Pricing, 2014, Sec. 11.6).

The particle method calibrates the leverage factor forward in time on one set of simulated paths,
the particles. At t_0 = 0 every particle sits at the forward and L(0, s) = sigma_Dup(0, s) /
sqrt(Y_0). At each later node t_j, given the particles (S^k_j, Y^k_j), it

1. estimates the conditional mean m_j(s) = E[Y_{t_j} | S_{t_j} = s] at N_j nodes s_i by the
   Nadaraya-Watson regression with the quartic kernel and bandwidth h_j;
2. sets L(t_j, s_i) = sigma_Dup(t_j, log s_i) / sqrt(m_j(s_i)) and interpolates in s by the
   not-a-knot cubic spline, constant beyond the outer nodes;
3. advances the particles by one Euler step with this leverage factor.

The settings of the book: the nodes are N_j = max(ceil(30 sqrt(t_j)), 15) equally spaced forward
levels between the digital quantiles of the target at the probabilities 0.001 and 0.999, and the
bandwidth is h_j = 1.5 sigma_VS(t_j) sqrt(max(t_j, 1/4)) N^(-1/5), with N the number of particles
and sigma_VS(t) = sqrt(int_0^t xi(s) ds / t) the variance-swap volatility of the target. Between
the nodes of the time grid, the leverage factor is that of the last node.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.functions import compiled
from locsvol.regression.cubic import cubic_minimum, flat_cubic, not_a_knot
from locsvol.regression.kernel import kernel_means
from locsvol.simulation.euler import euler_step


@partial(jax.tree_util.register_dataclass, data_fields=["target", "variance"], meta_fields=[])
@dataclass(frozen=True)
class _Origin:
    """L(0, s) = sigma_Dup(0, s) / sqrt(Y_0)."""

    target: object
    variance: jax.Array

    def __call__(self, t, x):
        return self.target(jnp.zeros_like(t), x) / jnp.sqrt(self.variance)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["nodes", "values", "coefficients", "count"],
    meta_fields=[],
)
@dataclass(frozen=True)
class _Slice:
    """The cubic in s = exp(x) of one time node."""

    nodes: jax.Array
    values: jax.Array
    coefficients: jax.Array
    count: jax.Array

    def __call__(self, t, x):
        return flat_cubic(self.nodes, self.values, self.coefficients, self.count, jnp.exp(x))


_advance = jax.jit(euler_step)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["times", "nodes", "values", "coefficients", "counts", "target", "origin_variance"],
    meta_fields=[],
)
@dataclass(frozen=True)
class ParticleLeverage:
    """The leverage factor of the particle method: at node j, the cubic through (nodes[j],
    values[j]); at the first node, sigma_Dup(0, x) / sqrt(origin_variance)."""

    times: jax.Array
    nodes: jax.Array
    values: jax.Array
    coefficients: jax.Array
    counts: jax.Array
    target: object
    origin_variance: jax.Array

    def __call__(self, t, x):
        t, x = jnp.broadcast_arrays(jnp.asarray(t), jnp.asarray(x))
        index = jnp.clip(jnp.searchsorted(self.times, t, side="right") - 1, 0, self.times.size - 2)
        flat = t.size
        spline = jax.vmap(flat_cubic)(
            self.nodes[index].reshape(flat, -1),
            self.values[index].reshape(flat, -1),
            self.coefficients[index].reshape(flat, *self.coefficients.shape[1:]),
            self.counts[index].reshape(-1),
            jnp.exp(x).reshape(-1),
        ).reshape(t.shape)

        def with_origin():
            origin = self.target(jnp.zeros_like(t), x) / jnp.sqrt(self.origin_variance)
            return jnp.where(index == 0, origin, spline)

        value = jax.lax.cond(jnp.any(index == 0), with_origin, lambda: spline)
        value = jnp.where(jnp.isfinite(t) & (t >= 0) & jnp.isfinite(x), value, jnp.nan)
        return jnp.where((t >= 0) & (t <= self.times[-1]) & jnp.isfinite(t), value, jnp.nan)


def variance_swap_volatility(forward_variance, times):
    """sigma_VS(t) = sqrt(int_0^t xi(s) ds / t), and sqrt(xi(0)) at t = 0, for a piecewise
    constant forward variance (boundaries, values) with the last value extended."""
    boundaries, values = forward_variance.boundaries, forward_variance.values
    left = jnp.concatenate((jnp.zeros(1), boundaries[:-1]))
    prefix = jnp.concatenate((jnp.zeros(1), jnp.cumsum(jnp.diff(left) * values[:-1])))
    index = jnp.minimum(jnp.searchsorted(boundaries, times, side="right"), values.size - 1)
    total = prefix[index] + (times - left[index]) * values[index]
    first = values[jnp.minimum(jnp.searchsorted(boundaries, 0.0, side="right"), values.size - 1)]
    return jnp.sqrt(jnp.where(times == 0, first, total / jnp.where(times > 0, times, 1.0)))


def particle_design(
    times,
    bounds,
    variance_swap,
    particles,
    kappa=1.5,
    minimum_time=0.25,
    nodes_per_sqrt_year=30.0,
    minimum_nodes=15,
):
    """Node counts, node levels (padded) and bandwidths at the nodes t_0, ..., t_{n-1}.

    bounds (n, 2) holds the forward levels of the 0.001 and 0.999 digital quantiles; the first row
    is (1, 1).
    """
    t = times[:-1]
    counts = (
        jnp.ceil(jnp.maximum(nodes_per_sqrt_year * jnp.sqrt(t), minimum_nodes))
        .astype(jnp.int32)
        .at[0]
        .set(1)
    )
    capacity = max(minimum_nodes, int(jnp.max(counts)))
    fractions = jnp.minimum(
        jnp.arange(capacity)[None, :] / jnp.maximum(counts[:, None] - 1, 1), 1.0
    )
    nodes = bounds[:, :1] + (bounds[:, 1:] - bounds[:, :1]) * fractions
    bandwidths = (
        kappa * variance_swap * jnp.sqrt(jnp.maximum(t, minimum_time)) * particles ** (-0.2)
    )
    return counts, nodes, bandwidths


def particle_method(target, paths, bounds, variance_swap, kernel="quartic", block_rows=4096):
    """The leverage factor of the particle method on the particles of paths.

    target is the local volatility sigma_Dup(t, x), paths the simulated variance factor with its
    spot increments, bounds the digital-quantile levels (n, 2) and variance_swap sigma_VS at the
    nodes t_0, ..., t_{n-1}.
    """
    target = compiled(target)
    times, variance = paths.times, paths.variance
    particles = variance.shape[0]
    counts, nodes, bandwidths = particle_design(
        times, bounds, variance_swap, jnp.asarray(particles, jnp.float64)
    )
    capacity = nodes.shape[1]
    x = jnp.zeros(particles)
    origin_variance = variance[0, 0]
    tables = []
    for j in range(times.size - 1):
        y = variance[:, j]
        if j == 0:
            value = target(0.0, 0.0) / jnp.sqrt(y[0])
            values = jnp.full(capacity, value)
            coefficients = jnp.zeros((capacity - 1, 4))
        else:
            means = kernel_means(jnp.exp(x), y, nodes[j], bandwidths[j], kernel, block_rows)
            if not bool(jnp.all(jnp.isfinite(means) & (means >= 1e-12))):
                raise ValueError(f"no conditional mean at some node of time step {j}")
            values = target(times[j], jnp.log(nodes[j])) / jnp.sqrt(means)
            coefficients = not_a_knot(nodes[j], values, counts[j])
            if not bool(cubic_minimum(nodes[j], coefficients, counts[j]) >= 0):
                raise ValueError(f"negative leverage factor at time step {j}")
            values = jnp.where(jnp.arange(capacity) < counts[j], values, values[counts[j] - 1])
        tables.append((values, coefficients))
        leverage = (
            _Origin(target, origin_variance)
            if j == 0
            else _Slice(nodes[j], values, coefficients, counts[j])
        )
        x = _advance(leverage, times[j], times[j + 1] - times[j], x, y, paths.spot_increments[:, j])
        if not bool(jnp.all(jnp.isfinite(jnp.exp(x)))):
            raise ValueError(f"particles left the finite range at time step {j}")
    values, coefficients = (jnp.stack(v) for v in zip(*tables, strict=True))
    return ParticleLeverage(times, nodes, values, coefficients, counts, target, origin_variance)
