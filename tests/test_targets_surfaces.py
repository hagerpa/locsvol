"""Raw SVI slices, the SSVI and refined SVI surfaces, the price inversion and Dupire's formula."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import locsvol  # noqa: F401
from locsvol.pricing.black import black_price
from locsvol.targets import SSVI, Dupire, RefinedSVI
from locsvol.targets.dupire import butterfly_factor, local_variance
from locsvol.targets.inversion import otm_price, otm_total_variance
from locsvol.targets.ssvi import ssvi_total_variance
from locsvol.targets.svi import (
    svi,
    svi_derivatives,
    svi_from_ssvi,
    svi_minimum,
    svi_valid,
    svi_wings,
)

RHO, ETA, GAMMA = -0.6, 1.0, 0.4
EXPIRIES = jnp.asarray([0.1, 0.5, 1.0])
THETA = jnp.asarray([0.004, 0.018, 0.04])
K = jnp.linspace(-1.5, 1.5, 61)


@pytest.fixture(scope="module")
def ssvi():
    return SSVI(
        jnp.concatenate((jnp.zeros(1), EXPIRIES)),
        jnp.concatenate((jnp.zeros(1), THETA)),
        RHO,
        ETA,
        GAMMA,
    )


@pytest.fixture(scope="module")
def refined():
    phi = ETA * THETA**-GAMMA
    slices = jnp.stack([svi_from_ssvi(theta, RHO, p) for theta, p in zip(THETA, phi, strict=True)])
    return RefinedSVI(EXPIRIES, slices, shift=0.02, level=0.3, long_forward_variance=0.05)


def paper_ssvi(theta, k):
    phi = ETA * theta**-GAMMA
    return theta / 2 * (1 + RHO * phi * k + np.sqrt((phi * k + RHO) ** 2 + 1 - RHO**2))


def test_svi_slice_formula_derivatives_minimum_and_wings():
    p = jnp.asarray([0.01, 0.1, -0.5, 0.05, 0.2])
    a, b, rho, m, sigma = np.asarray(p)
    k = np.linspace(-30.0, 30.0, 601)
    direct = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
    np.testing.assert_allclose(svi(p, k), direct, rtol=1e-13)
    first, second = svi_derivatives(p, k)
    np.testing.assert_allclose(
        first, jax.vmap(jax.grad(svi, argnums=1), (None, 0))(p, k), rtol=1e-12
    )
    np.testing.assert_allclose(second, b * sigma**2 / ((k - m) ** 2 + sigma**2) ** 1.5, rtol=1e-12)
    fine = np.linspace(-1.0, 1.0, 200001)
    assert abs(float(svi_minimum(p)) - float(jnp.min(svi(p, fine)))) < 1e-12
    left, right = svi_wings(p)
    np.testing.assert_allclose(svi(p, 1e6) / 1e6, right, rtol=1e-6)
    np.testing.assert_allclose(svi(p, -1e6) / 1e6, left, rtol=1e-6)
    assert bool(svi_valid(p)) and not bool(svi_valid(p.at[0].set(-1.0)))


def test_svi_slice_of_ssvi_is_the_ssvi_smile():
    for theta in (0.004, 0.03):
        phi = ETA * theta**-GAMMA
        p = svi_from_ssvi(theta, RHO, phi)
        np.testing.assert_allclose(svi(p, K), paper_ssvi(theta, np.asarray(K)), rtol=1e-13)
        np.testing.assert_allclose(
            svi(p, K), ssvi_total_variance(theta, K, RHO, ETA, GAMMA), rtol=1e-13
        )


def test_ssvi_surface_values_jet_and_conditions(ssvi):
    t = jnp.asarray([0.05, 0.3, 0.75])[:, None]
    theta = np.interp(
        np.asarray(t), np.concatenate(([0.0], EXPIRIES)), np.concatenate(([0.0], THETA))
    )
    np.testing.assert_allclose(
        ssvi.total_variance(t, K), paper_ssvi(theta, np.asarray(K)), rtol=1e-13
    )
    np.testing.assert_allclose(ssvi.total_variance(EXPIRIES, 0.0), THETA, rtol=1e-14)
    assert float(ssvi.total_variance(0.0, 0.3)) == 0.0
    assert np.isnan(ssvi.total_variance(1.01, 0.0)) and np.isnan(ssvi.total_variance(0.5, 25.0))
    # Deep in the right wing the two halves of w cancel; the rationalized form keeps precision.
    far = jnp.asarray([5.0, 10.0, 19.0])
    np.testing.assert_allclose(
        ssvi.total_variance(0.1, far), paper_ssvi(0.004, np.asarray(far)), rtol=1e-9
    )
    w, w_t, w_k, w_kk = ssvi.jet(t, K)
    grad_k = jax.vmap(jax.vmap(jax.grad(ssvi.total_variance, argnums=1), (None, 0)), (0, None))
    np.testing.assert_allclose(w_k, grad_k(t[:, 0], K), rtol=1e-11, atol=1e-15)
    hess_k = jax.vmap(jax.grad(jax.grad(ssvi.total_variance, argnums=1), argnums=1), (None, 0))
    np.testing.assert_allclose(w_kk[1], hess_k(0.3, K), rtol=1e-10)
    h = 1e-6
    finite = (ssvi.total_variance(t + h, K) - ssvi.total_variance(t, K)) / h
    np.testing.assert_allclose(w_t, finite, rtol=1e-5)
    # At a knot the time derivative is the right one.
    _, right, _, _ = ssvi.jet(0.1, 0.2)
    later = (ssvi.total_variance(0.1 + h, 0.2) - ssvi.total_variance(0.1, 0.2)) / h
    np.testing.assert_allclose(right, later, rtol=1e-5)
    theta_n = float(THETA[-1])
    phi = ETA * theta_n**-GAMMA
    expected = [4 - theta_n * phi * (1 + RHO), 4 - theta_n * phi * (1 - RHO)]
    expected += [4 - theta_n * phi**2 * (1 + RHO), 4 - theta_n * phi**2 * (1 - RHO)]
    np.testing.assert_allclose(ssvi.conditions(), expected, rtol=1e-14)


def test_ssvi_theta_extension(ssvi):
    extended = ssvi.extended(1.5)
    np.testing.assert_array_equal(extended.theta_times[:-1], ssvi.theta_times)
    slope = (THETA[-1] - THETA[-2]) / (EXPIRIES[-1] - EXPIRIES[-2])
    np.testing.assert_allclose(extended.theta_values[-1], THETA[-1] + 0.5 * slope, rtol=1e-15)
    assert ssvi.extended(0.8) is ssvi


def test_inversion_of_the_otm_price_and_its_derivatives():
    k = jnp.asarray([-0.4, -0.05, 0.0, 0.1, 0.6])
    w = jnp.asarray([0.03, 0.01, 0.02, 0.015, 0.05])
    lower, upper = 0.5 * w, 2.0 * w
    price = otm_price(w, k)
    inverted = otm_total_variance(price, k, lower, upper, otm_price(lower, k), otm_price(upper, k))
    np.testing.assert_allclose(inverted, w, rtol=1e-12)

    def inverse(p, x, w0):
        return otm_total_variance(
            p, x, 0.5 * w0, 2.0 * w0, otm_price(0.5 * w0, x), otm_price(2.0 * w0, x)
        )

    dp = jax.vmap(jax.grad(inverse, argnums=0))(price, k, w)
    vega = jax.vmap(jax.grad(black_price, argnums=0))(w, k, k >= 0)
    np.testing.assert_allclose(dp, 1.0 / vega, rtol=1e-10)
    # The variance of a price that moves with k along a fixed smile does not move.
    along = jax.vmap(jax.grad(lambda x, w0: inverse(otm_price(w0, x), x, w0)))(k, w)
    np.testing.assert_allclose(along, 0.0, atol=1e-9)
    outside = otm_total_variance(
        2.0 * otm_price(upper, k), k, lower, upper, otm_price(lower, k), otm_price(upper, k)
    )
    assert np.all(np.isnan(outside))


def test_refined_surface_slices_extensions_and_interpolation(refined):
    for i, expiry in enumerate(EXPIRIES):
        np.testing.assert_allclose(
            refined.total_variance(expiry, K), svi(refined.slices[i], K), rtol=1e-15
        )
    t, t0, c, first = 0.03, 0.02, 0.3, float(EXPIRIES[0])
    x = t / first
    expected = x * (1 + c * (1 - x)) * svi(refined.slices[0], K * np.sqrt((first + t0) / (t + t0)))
    np.testing.assert_allclose(refined.total_variance(t, K), expected, rtol=1e-14)
    expected = svi(refined.slices[-1], K) + 0.05 * 0.1
    np.testing.assert_allclose(refined.total_variance(1.1, K), expected, rtol=1e-14)
    assert np.all(refined.total_variance(0.0, K) == 0.0)
    assert np.isnan(refined.total_variance(1.3, 0.0)) and np.isnan(
        refined.total_variance(0.5, 21.0)
    )
    # Between expiries the out-of-the-money prices interpolate in the square root of theta.
    t = 0.3
    theta = np.interp(t, np.asarray(EXPIRIES), np.asarray(refined.atm()))
    alpha = (np.sqrt(refined.atm()[1]) - np.sqrt(theta)) / (
        np.sqrt(refined.atm()[1]) - np.sqrt(refined.atm()[0])
    )
    prices = alpha * otm_price(svi(refined.slices[0], K), K) + (1 - alpha) * otm_price(
        svi(refined.slices[1], K), K
    )
    np.testing.assert_allclose(otm_price(refined.total_variance(t, K), K), prices, rtol=1e-10)


def test_refined_jet_matches_differences(refined):
    t = jnp.asarray([0.05, 0.3, 0.75, 1.1])[:, None]
    k = jnp.linspace(-0.8, 0.8, 17)[None, :]
    w, w_t, w_k, w_kk = refined.jet(t, k)
    np.testing.assert_allclose(w, refined.total_variance(t, k), rtol=1e-14)
    h = 1e-6
    np.testing.assert_allclose(w_t, (refined.total_variance(t + h, k) - w) / h, rtol=1e-4)
    np.testing.assert_allclose(
        w_k,
        (refined.total_variance(t, k + h) - refined.total_variance(t, k - h)) / (2 * h),
        rtol=1e-6,
    )
    h = 1e-4
    second = (refined.total_variance(t, k + h) - 2 * w + refined.total_variance(t, k - h)) / h**2
    np.testing.assert_allclose(w_kk, second, rtol=1e-4, atol=1e-6)
    assert np.all(np.isnan(np.asarray(refined.jet(1.2, 0.0))))


def test_butterfly_factor_is_the_paper_denominator():
    w, x, w_k, w_kk = 0.04, jnp.linspace(-1, 1, 11), -0.03, 0.2
    paper = 1 - x / w * w_k + 0.25 * (-0.25 - 1 / w + x**2 / w**2) * w_k**2 + 0.5 * w_kk
    np.testing.assert_allclose(butterfly_factor(w, x, w_k, w_kk), paper, rtol=1e-13)
    assert np.isnan(local_variance(0.04, -0.01, 0.0, 0.0, 0.0))


@partial(jax.tree_util.register_dataclass, data_fields=["variance"], meta_fields=[])
@dataclass(frozen=True)
class Flat:
    """w(t, k) = sigma^2 t on |k| <= 5 as a surface."""

    variance: float
    log_moneyness_bounds = (-5.0, 5.0)

    def jet(self, t, k):
        t, k = jnp.broadcast_arrays(t, k)
        inside = (t > 0) & (jnp.abs(k) <= 5.0)
        values = (self.variance * t, self.variance + 0 * t, 0 * t, 0 * t)
        return tuple(jnp.where(inside, v, jnp.nan) for v in values)

    def initial_local_variance(self, k):
        return jnp.where(jnp.abs(k) <= 5.0, self.variance, jnp.nan)


def test_dupire_of_a_flat_surface_and_its_domain():
    dupire = Dupire(Flat(0.04))
    t, x = jnp.asarray([0.0, 0.1, 1.0])[:, None], jnp.asarray([-4.0, 0.0, 4.0, 6.0])[None, :]
    sigma = np.asarray(dupire(t, x))
    np.testing.assert_allclose(sigma[:, :3], 0.2, rtol=1e-15)
    assert np.all(np.isnan(sigma[:, 3]))
    assert dupire.log_moneyness_bounds == (-5.0, 5.0)


def test_dupire_limits_at_time_zero(ssvi, refined):
    # SSVI: theta_1 / T_1 at the money for gamma < 1/2, nan elsewhere at t = 0.
    np.testing.assert_allclose(
        Dupire(ssvi).local_variance(0.0, 0.0), THETA[0] / EXPIRIES[0], rtol=1e-15
    )
    assert np.isnan(Dupire(ssvi)(0.0, 0.1))
    # Refined with a shift: the whole row at t = 0 is the limit of Dupire's local variance.
    x = jnp.linspace(-0.5, 0.5, 11)
    row = Dupire(refined).local_variance(0.0, x)
    near = Dupire(refined).local_variance(1e-9, x)
    np.testing.assert_allclose(row, near, rtol=1e-6)
    self_similar = RefinedSVI(refined.expiries, refined.slices)
    assert np.isfinite(Dupire(self_similar)(0.0, 0.0)) and np.isnan(Dupire(self_similar)(0.0, 0.1))
    np.testing.assert_allclose(
        Dupire(self_similar).local_variance(0.0, 0.0),
        Dupire(self_similar).local_variance(1e-9, 0.0),
        rtol=1e-4,
    )
