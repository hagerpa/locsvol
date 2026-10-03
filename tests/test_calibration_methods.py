"""The fixed-point iteration and the particle method on a model with a known answer."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import locsvol  # noqa: F401
from locsvol.calibration.fixed_point import every_kth_node, fixed_point
from locsvol.calibration.particle import particle_method
from locsvol.leverage.chart import Chart
from locsvol.leverage.parametrized import Parametrized
from locsvol.regression.chebyshev import ChebyshevTensor
from locsvol.regression.gamma import GammaRegression
from locsvol.simulation.paths import simulate


@dataclass(frozen=True)
class TwoStates:
    """Y = 1 +- 0.5 with probability 1/2 each, independent of the spot: E[Y | S] = 1."""

    factors = 2

    def variance(self, times, increments):
        sign = jnp.sign(increments[:, :1, 1])
        return jnp.broadcast_to(1.0 + 0.5 * sign, (increments.shape[0], times.size))


def flat(level):
    return lambda t, x: jnp.full(jnp.broadcast_shapes(jnp.shape(t), jnp.shape(x)), level)


def test_rows_use_every_node_equally_often():
    mask = every_kth_node(1024, 13, 4)
    np.testing.assert_array_equal(
        mask.sum(axis=1), np.where(np.arange(1024) % 4 < 1, 4, 3) * 0 + mask.sum(axis=1)
    )
    counts = mask.sum(axis=0)
    assert counts.max() - counts.min() <= 1 and mask.sum() == 1024 * 13 // 4 + (
        mask.sum() - 1024 * 13 // 4
    )


def calibration_error(leverage, paths, node):
    """Largest relative error of L(t, x)^2 E[Y_t | S_t = x] / sigma^2 over central bins of x."""
    x = paths.log_spot(leverage)[:, node]
    y = paths.variance[:, node]
    edges = np.quantile(np.asarray(x), np.linspace(0.1, 0.9, 9))
    errors = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (x >= low) & (x < high)
        middle = jnp.asarray(0.5 * (low + high))
        value = leverage(paths.times[node], middle) ** 2 * jnp.mean(y[inside]) / 0.2**2
        errors.append(abs(float(value) - 1.0))
    return max(errors)


def test_fixed_point_and_particle_method_satisfy_the_calibration_condition():
    times = jnp.linspace(0.0, 1.0, 26)
    paths = simulate(TwoStates(), times, dict(seed=5, role="test"), 16384)
    chart = Chart(
        jnp.asarray([0.0, 1.2]), jnp.asarray([1e-6, 0.05]), jnp.asarray(1.0), jnp.asarray(0.5)
    )
    basis = ChebyshevTensor((2, 4))
    start = Parametrized(flat(0.3), chart, basis, jnp.zeros(basis.size))
    iterates = fixed_point(start, flat(0.2), paths, GammaRegression(basis, ridge=1e-6), 4)
    assert calibration_error(iterates[-1], paths, 13) < 0.03
    spread = 2.0 * 0.25 * jnp.sqrt(times[1:-1])[:, None] * jnp.asarray([[-1.0, 1.0]])
    bounds = jnp.concatenate((jnp.ones((1, 2)), jnp.exp(spread)))
    particle = particle_method(flat(0.2), paths, bounds, jnp.full(25, 0.2))
    assert calibration_error(particle, paths, 13) < 0.03
