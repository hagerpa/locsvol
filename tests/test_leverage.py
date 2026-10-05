"""Coordinates, tables and the parametrized leverage factor."""

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtri

import locsvol  # noqa: F401
from locsvol.leverage.chart import Chart, quantile_envelope
from locsvol.leverage.parametrized import Parametrized
from locsvol.leverage.table import Capped, Table, table_times, tabulate
from locsvol.regression.chebyshev import ChebyshevTensor
from locsvol.targets.analytic import ConstantVolatility


def chart(**changes):
    times = jnp.asarray([0.0, 0.5, 1.2])
    return Chart(times, 0.04 * times + 1e-6, jnp.asarray(1.4), jnp.asarray(0.5), **changes)


def test_state_coordinate_inverts_and_time_coordinate_spans_the_interval():
    c = chart()
    t = jnp.asarray([0.01, 0.3, 1.0])
    x = jnp.asarray([-0.4, 0.0, 0.25])
    np.testing.assert_allclose(c.log_moneyness(t, c(t, x)[:, 1]), x, atol=1e-14)
    np.testing.assert_allclose(c.time(jnp.asarray([0.0, 1.1])), [-1.0, 1.0], atol=1e-14)
    clamped = chart(time_clamp=0.2)
    np.testing.assert_allclose(clamped.time(0.05), clamped.time(0.2))


def test_envelope_clips_the_state_coordinate_and_maps_it_onto_the_interval():
    c = chart()
    times = jnp.asarray([0.0, 0.5, 1.0])
    lower, upper = jnp.asarray([0.0, -0.8, -0.9]), jnp.asarray([0.0, 0.6, 0.7])
    stretched = c.with_envelope(times, lower, upper)
    t = jnp.full(5, 0.5)
    u = jnp.asarray([-0.95, -0.8, -0.1, 0.6, 0.9])
    x = c.log_moneyness(t, u)
    np.testing.assert_allclose(
        stretched(t, x)[:, 1], [-1.0, -1.0, 2 * (-0.1 + 0.8) / 1.4 - 1, 1.0, 1.0], atol=1e-13
    )
    np.testing.assert_array_equal(stretched(jnp.zeros(2), jnp.asarray([-0.1, 0.1]))[:, 1], 0.0)


def test_quantile_envelope_of_a_lognormal_law():
    c = chart()
    sigma, level = 0.2, 1e-3
    times = jnp.asarray([0.0, 0.1, 0.5, 1.0])
    _, lower, upper = quantile_envelope(c, ConstantVolatility(sigma), times, level)
    w = sigma**2 * times[1:]
    for edge, probability in ((lower, level), (upper, 1 - level)):
        x = -w / 2 + jnp.sqrt(w) * ndtri(probability)
        np.testing.assert_allclose(c.log_moneyness(times[1:], edge[1:]), x, atol=1e-8)
        assert float(edge[0]) == 0.0


def test_table_interpolates_linearly_and_caps():
    c = chart()

    def plane(t, x):
        return 0.2 + 0.1 * t + 0.05 * c(t, x)[..., 1]

    times = table_times([0.25, 0.75], 1.1, 1e-4, 0.05, 20, 0.1, (0.0,))
    table = tabulate(plane, c, times)
    t = jnp.asarray([0.01, 0.33, 0.9])
    x = jnp.asarray([-0.3, 0.05, 0.4])
    np.testing.assert_allclose(table(t, x), plane(t, x), rtol=1e-12)
    np.testing.assert_allclose(table(jnp.asarray(0.33), x), plane(jnp.asarray(0.33), x), rtol=1e-12)
    capped = Capped(plane, 0.21, -0.2, 0.2)
    x = jnp.asarray([-1.0, 0.1, 1.0])
    expected = jnp.minimum(plane(jnp.asarray(0.5), jnp.clip(x, -0.2, 0.2)), 0.21)
    np.testing.assert_allclose(capped(jnp.asarray(0.5), x), expected)


def test_table_takes_the_value_toward_the_money_where_the_function_has_none():
    c = chart()

    def gapped(t, x):
        value = 0.2 + 0.1 * t + 0.05 * c(t, x)[..., 1]
        return jnp.where(x > 0.3, jnp.nan, jnp.where(x < -0.4, -1.0, value))

    times = jnp.asarray([0.0, 0.5, 1.0])
    table = tabulate(gapped, c, times)
    values = np.asarray(table.values)
    x = np.asarray(c.log_moneyness(times[:, None], table.nodes[None, :]))
    assert np.all(np.isfinite(values)) and np.all(values > 0)
    for row in range(times.size):
        inside = np.flatnonzero((x[row] >= -0.4) & (x[row] <= 0.3))
        np.testing.assert_array_equal(values[row, x[row] > 0.3], values[row, inside[-1]])
        np.testing.assert_array_equal(values[row, x[row] < -0.4], values[row, inside[0]])


def test_parametrized_leverage_at_one_time_equals_the_vector_evaluation():
    c = chart()
    basis = ChebyshevTensor((2, 3))
    base = Table(jnp.asarray([0.0, 1.2]), jnp.linspace(-1, 1, 5), jnp.full((2, 5), 0.2), c)
    beta = jnp.asarray(np.random.default_rng(1).normal(0.0, 0.1, basis.size))
    leverage = Parametrized(base, c, basis, beta)
    x = jnp.linspace(-0.5, 0.5, 7)
    np.testing.assert_allclose(
        leverage(jnp.asarray(0.4), x), leverage(jnp.full(7, 0.4), x), rtol=1e-13
    )
    np.testing.assert_allclose(Parametrized(base, c, basis, 0 * beta)(0.4, x), 0.2, rtol=1e-14)
    t = jnp.linspace(0.01, 1.0, 7)
    np.testing.assert_allclose(
        leverage.log_factor(t, x), basis(c(t, x)) @ beta, rtol=1e-13, atol=1e-15
    )
    np.testing.assert_allclose(leverage.shifted(-beta).coefficients, 0.0)
