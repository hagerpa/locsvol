"""Random numbers, time grids and the Euler scheme."""

import jax.numpy as jnp
import numpy as np

import locsvol  # noqa: F401
from locsvol.simulation.euler import log_spot
from locsvol.simulation.grid import graded_times, trapezoid_weights, with_dates
from locsvol.simulation.random import (
    bridge_increments,
    bridge_schedule,
    brownian_increments,
    chronological_increments,
    sobol_normals,
    stream_digest,
)

STREAM = dict(seed=7, role="test")


def test_stream_digest_depends_on_every_entry():
    assert stream_digest(STREAM) == stream_digest(dict(role="test", seed=7))
    assert stream_digest(STREAM) != stream_digest(dict(STREAM, role="other"))


def test_bridge_ends_at_the_first_normal_and_has_brownian_variance():
    times = jnp.asarray([0.0, 0.1, 0.25, 0.5, 1.0])
    normals = sobol_normals(STREAM, 4096, 4, 1)
    bridge = bridge_increments(normals, times, bridge_schedule(times))
    np.testing.assert_allclose(bridge.sum(axis=1)[:, 0], normals[:, 0, 0], atol=1e-14)
    np.testing.assert_allclose(np.var(bridge[:, :, 0], axis=0), np.diff(times), rtol=0.06)
    chronological = chronological_increments(normals, times)
    np.testing.assert_allclose(np.var(chronological[:, :, 0], axis=0), np.diff(times), rtol=0.06)


def test_antithetic_pairs_and_reproducibility():
    times = jnp.linspace(0.0, 1.0, 9)
    first = brownian_increments(times, STREAM, 256, 2)
    np.testing.assert_array_equal(first[:128], -first[128:])
    np.testing.assert_array_equal(first, brownian_increments(times, STREAM, 256, 2))
    other = brownian_increments(times, dict(STREAM, seed=8), 256, 2)
    assert not np.array_equal(first, other)
    mc = brownian_increments(times, STREAM, 64, 2, sobol=False, bridge=False, antithetic=False)
    assert mc.shape == (64, 8, 2)


def test_graded_grid_with_dates_and_trapezoid_weights():
    times = graded_times(1.1, 90 / 365, 800, 2.0, 1.1 / 440)
    assert times[0] == 0.0 and times[-1] == 1.1 and bool(jnp.all(jnp.diff(times) > 0))
    np.testing.assert_allclose(times[800], 90 / 365)
    dated = with_dates(times, [0.3, 0.3, 1.0])
    assert dated.size == times.size + 2 and {0.3, 1.0} <= set(np.asarray(dated).tolist())
    np.testing.assert_allclose(jnp.sum(trapezoid_weights(dated)), 1.0, atol=1e-15)


def test_euler_scheme_is_a_martingale_with_the_exact_drift():
    times = jnp.linspace(0.0, 1.0, 51)
    sigma = 0.3
    dw = brownian_increments(times, STREAM, 65536, 1)[:, :, 0]

    def leverage(t, x):
        return jnp.full_like(x, sigma)

    x = log_spot(leverage, times, jnp.ones((65536, times.size)), dw)
    np.testing.assert_allclose(jnp.mean(jnp.exp(x[:, -1])), 1.0, atol=2e-3)
    flat = log_spot(leverage, times, jnp.ones((2, times.size)), jnp.zeros((2, 50)))
    np.testing.assert_allclose(flat[:, -1], -0.5 * sigma**2, rtol=1e-12)
