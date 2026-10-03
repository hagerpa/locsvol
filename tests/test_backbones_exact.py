"""The Bergomi and Heston backbones: exact transitions and normalized means."""

import jax.numpy as jnp
import numpy as np
from scipy.stats import ncx2

import locsvol  # noqa: F401
from locsvol.backbones.bergomi import Bergomi, floored_lognormal_mean
from locsvol.backbones.heston import Heston, noncentral_chi_square_cdf
from locsvol.simulation.random import brownian_increments

STREAM = dict(seed=11, role="test")


def test_bergomi_factors_have_the_exact_ou_variance_and_unit_mean():
    model = Bergomi(nu=1.5, theta=0.3, rho1=-0.7, rho2=-0.3)
    times = jnp.linspace(0.0, 1.0, 21)
    increments = brownian_increments(times, STREAM, 65536, 3)
    variance = model.variance(times, increments)
    np.testing.assert_allclose(variance[:, 0], 1.0, rtol=1e-12)
    np.testing.assert_allclose(jnp.mean(variance, axis=0), 1.0, atol=0.03)
    decay, innovation = model.transition(jnp.asarray(0.05))
    np.testing.assert_allclose(decay, jnp.exp(-jnp.array([4.0, 0.12]) * 0.05))
    np.testing.assert_allclose(jnp.sum(innovation[0] ** 2) * 0.05, (1 - jnp.exp(-0.4)) / 8.0)


def test_floored_lognormal_mean_matches_quadrature():
    mean, variance, floor = -0.5, 1.0, 0.2
    z = np.linspace(-12, 12, 200001)
    density = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    exact = np.trapezoid(np.maximum(np.exp(mean + z), floor) * density, z)
    np.testing.assert_allclose(floored_lognormal_mean(mean, variance, floor), exact, rtol=1e-9)


def test_heston_scheme_matches_the_cir_mean_and_chi_square_law():
    model = Heston(kappa=2.0, theta=0.04, sigma=0.6, rho=-0.7, v0=0.03)
    times = jnp.linspace(0.0, 1.0, 101)
    increments = brownian_increments(times, STREAM, 65536, 2)
    variance = model.variance(times, increments)
    np.testing.assert_allclose(jnp.mean(variance[:, -1]), 1.0, atol=0.02)
    x, degrees, noncentrality = jnp.asarray(3.0), jnp.asarray(2.5), jnp.asarray(4.0)
    np.testing.assert_allclose(
        noncentral_chi_square_cdf(x, degrees, noncentrality), ncx2.cdf(3.0, 2.5, 4.0), rtol=1e-10
    )
