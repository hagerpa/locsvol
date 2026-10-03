"""Saving and loading leverage factors, targets and backbones."""

import jax.numpy as jnp
import numpy as np
import pytest

import locsvol
from locsvol.backbones import Bergomi
from locsvol.leverage import Chart, Parametrized, Table
from locsvol.regression import ChebyshevTensor


def test_a_parametrized_leverage_factor_round_trips(tmp_path):
    times = jnp.asarray([0.0, 0.5, 1.2])
    chart = Chart(times, 0.04 * times + 1e-6, jnp.asarray(1.4), jnp.asarray(0.5))
    enveloped = chart.with_envelope(times, -0.8 * jnp.ones(3), 0.8 * jnp.ones(3), time_clamp=1e-3)
    base = Table(jnp.asarray([0.0, 1.2]), jnp.linspace(-1, 1, 5), jnp.full((2, 5), 0.2), chart)
    basis = ChebyshevTensor((2, 3))
    beta = jnp.asarray(np.random.default_rng(2).normal(0.0, 0.1, basis.size))
    leverage = Parametrized(base, enveloped, basis, beta)
    locsvol.save(tmp_path / "leverage.npz", leverage)
    loaded = locsvol.load(tmp_path / "leverage.npz")
    x = jnp.linspace(-0.5, 0.5, 7)
    np.testing.assert_array_equal(loaded(jnp.asarray(0.4), x), leverage(jnp.asarray(0.4), x))
    assert loaded.basis == basis and loaded.chart.time_clamp == 1e-3

    model = Bergomi(nu=2.32, theta=0.17, rho1=-0.75, rho2=-0.28)
    locsvol.save(tmp_path / "bergomi.npz", model)
    np.testing.assert_array_equal(
        locsvol.load(tmp_path / "bergomi.npz").mean(times), model.mean(times)
    )


def test_plain_functions_cannot_be_saved(tmp_path):
    times = jnp.asarray([0.0, 1.2])
    chart = Chart(times, jnp.asarray([1e-6, 0.05]), jnp.asarray(1.0), jnp.asarray(0.5))
    leverage = Parametrized(lambda t, x: 0.2 + 0 * x, chart, ChebyshevTensor((1, 1)), jnp.zeros(4))
    with pytest.raises(TypeError):
        locsvol.save(tmp_path / "plain.npz", leverage)
