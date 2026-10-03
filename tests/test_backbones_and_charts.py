"""The SABR, constant and custom backbones; affine, bounded and custom coordinates."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

import locsvol  # noqa: F401
from locsvol.backbones.constant import Constant
from locsvol.backbones.custom import CustomBackbone
from locsvol.backbones.sabr import SABR
from locsvol.leverage.coordinates import Affine, Bounded, CustomChart
from locsvol.simulation.paths import simulate
from locsvol.simulation.random import brownian_increments

TIMES = jnp.linspace(0.0, 1.0, 9)


def test_sabr_volatility_is_lognormal_and_the_variance_factor_has_mean_one():
    model = SABR(0.3, -0.5, 0.6)
    paths = simulate(model, TIMES, {"seed": 2}, 2**15)
    y = np.asarray(paths.variance)
    mean = np.asarray(model.mean(TIMES))
    np.testing.assert_allclose(mean, 0.09 * np.exp(0.36 * TIMES), rtol=1e-14)
    np.testing.assert_allclose(y.mean(axis=0), 1.0, atol=0.01)
    ell = 0.5 * np.log(y[:, -1] * mean[-1]) - np.log(0.3)
    np.testing.assert_allclose([ell.mean(), ell.var()], [-0.18, 0.36], atol=0.01)
    spot = np.asarray(paths.spot_increments).sum(axis=1)
    assert abs(np.corrcoef(ell, spot)[0, 1] + 0.5) < 0.02


def test_sabr_floored_mean_is_the_expectation_of_the_floored_variance():
    model = SABR(0.3, 0.2, 0.8, floor=0.25)
    for t in (0.0, 0.3, 1.5):
        if t == 0.0:
            expected = 0.3**2
        else:
            scale = 0.8 * np.sqrt(t)
            kink = (np.log(0.25 / 0.3) + 0.32 * t) / scale

            def integrand(z, scale=scale, t=t):
                return max(0.3 * np.exp(scale * z - 0.32 * t), 0.25) ** 2 * stats.norm.pdf(z)

            expected = integrate.quad(integrand, -12, 12, points=[kink], epsabs=0, epsrel=1e-12)[0]
        np.testing.assert_allclose(model.mean(t), expected, rtol=1e-10)


def test_custom_backbone_of_the_sabr_scheme():
    sabr = SABR(0.3, -0.5, 0.6, 0.05)

    def start(p, paths):
        return jnp.zeros(paths)

    def transition(p, state, t, dt, dw):
        return SABR(*p).step(state, dt, dw)

    def state_variance(p, state, t):
        return SABR(*p).raw_variance(state)

    def mean(p, t):
        return SABR(*p).mean(t)

    increments = brownian_increments(TIMES, {"seed": 4}, 256, 2)
    p = jnp.array([0.3, -0.5, 0.6, 0.05])
    custom = CustomBackbone(2, start, transition, state_variance, mean, p)
    np.testing.assert_allclose(
        custom.variance(TIMES, increments), sabr.variance(TIMES, increments), rtol=1e-14
    )
    unnormalized = CustomBackbone(2, start, transition, state_variance, parameters=p)
    np.testing.assert_allclose(
        unnormalized.variance(TIMES, increments),
        sabr.variance(TIMES, increments) * sabr.mean(TIMES),
        rtol=1e-14,
    )


def test_constant_backbone_gives_the_local_volatility_model():
    increments = brownian_increments(TIMES, {"seed": 4}, 8, 1)
    np.testing.assert_array_equal(Constant(0.2).variance(TIMES, increments), np.ones((8, 9)))


def test_affine_coordinates_and_their_inverse():
    chart = Affine((0.1, 0.05), (0.5, 0.2))
    z = chart(0.3, jnp.array([-0.1, 0.25]))
    np.testing.assert_allclose(z, [[0.4, -0.75], [0.4, 1.0]], rtol=1e-14)
    np.testing.assert_allclose(chart.log_moneyness(0.3, z[:, 1]), [-0.1, 0.25], rtol=1e-14)
    assert Affine((0.0, 0.0), (1.0, 1.0), with_time=False)(0.3, jnp.zeros(4)).shape == (4, 1)
    assert np.all(np.isnan(chart(-0.1, jnp.zeros(2))))
    assert np.all(np.isnan(chart(0.1, jnp.array([jnp.nan, jnp.inf]))))


def test_bounded_coordinates_clamp_or_reject_the_time():
    clamp = Bounded(0.05, 0.3, (0.1, 0.5))
    reject = Bounded(0.05, 0.3, (0.1, 0.5), "reject")
    x = jnp.array([-0.4, 0.05, 0.6])
    u = 2 / np.pi * np.arctan((np.asarray(x) - 0.05) / 0.3)
    np.testing.assert_allclose(clamp(0.3, x), np.stack([np.zeros(3), u], axis=1), atol=1e-15)
    np.testing.assert_allclose(clamp(0.7, x)[:, 0], 1.0)
    assert np.all(np.isnan(reject(0.7, x))) and np.all(np.isfinite(reject(0.3, x)))
    np.testing.assert_allclose(clamp.log_moneyness(0.3, u), x, rtol=1e-14, atol=1e-16)
    assert Bounded(0.0, 1.0)(0.3, x).shape == (3, 1)
    with pytest.raises(ValueError):
        Bounded(0.0, 1.0, (0.0, 1.0), "cut")


def test_custom_chart_is_undefined_outside_its_domain_and_inverts_by_bisection():
    def coordinates(t, x):
        return jnp.stack([t / 0.5, jnp.arcsinh(x / 0.2)], axis=-1)

    chart = CustomChart(coordinates, (0.0, 0.5), (-5.0, 5.0))
    x = jnp.linspace(-2.0, 2.0, 9)
    z = chart(0.25, x)
    np.testing.assert_allclose(z[:, 1], np.arcsinh(np.asarray(x) / 0.2), rtol=1e-14)
    np.testing.assert_allclose(chart.log_moneyness(0.25, z[:, 1]), x, atol=1e-12)
    assert np.all(np.isnan(chart(0.6, x))) and np.all(np.isnan(chart(0.25, jnp.array([6.0]))))
