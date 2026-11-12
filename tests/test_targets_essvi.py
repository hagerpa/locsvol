"""The extended SSVI surface: formula, derivatives, limits and Dupire's local volatility."""

import jax.numpy as jnp
import numpy as np

import locsvol  # noqa: F401
from locsvol.targets import ESSVI, SSVI, Dupire

# SPX, 23 June 2020 (Farkas, Ferrari and Ulrych, 2022, Tables 7 and 8)
TENORS = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
ATM = np.array([0.2652, 0.2717, 0.2545, 0.2381, 0.2298])
ETA, LAM, RHO_M, RHO_0, A = 0.9664, 0.4963, -0.7633, -0.2269, 340.68


def surface(**changes):
    parameters = dict(rho_0=RHO_0, rho_m=RHO_M, a=A, eta=ETA, lam=LAM)
    parameters.update(changes)
    return ESSVI(
        jnp.asarray(np.concatenate(([0.0], TENORS))),
        jnp.asarray(np.concatenate(([0.0], ATM**2 * TENORS))),
        **parameters,
    )


def paper_essvi(theta, k):
    phi = ETA * theta**-LAM * (1 + theta) ** (LAM - 1)
    rho = RHO_M + (RHO_0 - RHO_M) * np.exp(-A * theta)
    return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))


def test_total_variance_matches_the_published_formula_and_vanishes_at_zero():
    s = surface()
    k = np.linspace(-1.5, 1.5, 61)
    for tenor, atm in zip(TENORS, ATM, strict=True):
        np.testing.assert_allclose(
            s.total_variance(tenor, k), paper_essvi(atm**2 * tenor, k), rtol=1e-13
        )
        np.testing.assert_allclose(s.implied_volatility(tenor, 0.0), atm, rtol=1e-13)
    assert s.total_variance(0.0, 0.3) == 0.0
    assert np.isnan(s.total_variance(3.5, 0.0)) and np.isnan(s.total_variance(1.0, 25.0))
    assert np.array_equal(s.expiries, TENORS)


def test_jet_agrees_with_finite_differences_including_the_far_wings():
    s = surface()
    t = np.array([0.1, 0.3, 0.75, 1.5, 2.9])
    k = np.array([-12.0, -1.0, -0.2, 0.0, 0.15, 0.8, 12.0])
    tt, kk = np.meshgrid(t, k, indexing="ij")
    w, w_t, w_k, w_kk = (np.asarray(v) for v in s.jet(tt, kk))
    np.testing.assert_allclose(w, s.total_variance(tt, kk), rtol=1e-14)
    h = 1e-5
    w_plus, w_minus = (np.asarray(s.total_variance(tt + d, kk)) for d in (h, -h))
    np.testing.assert_allclose(w_t, (w_plus - w_minus) / (2 * h), rtol=1e-6)
    w_right, w_left = (np.asarray(s.total_variance(tt, kk + d)) for d in (h, -h))
    np.testing.assert_allclose(w_k, (w_right - w_left) / (2 * h), rtol=1e-6, atol=1e-10)
    h = 1e-3
    w_right, w_left = (np.asarray(s.total_variance(tt, kk + d)) for d in (h, -h))
    np.testing.assert_allclose(w_kk, (w_right - 2 * w + w_left) / h**2, rtol=1e-4, atol=1e-9)


def test_right_time_derivative_at_the_knots_and_extension():
    s = surface()
    _, w_t, _, _ = s.jet(1.0, 0.1)
    _, w_t_after, _, _ = s.jet(1.0 + 1e-9, 0.1)
    np.testing.assert_allclose(w_t, w_t_after, rtol=1e-6)
    assert np.isnan(s.jet(3.0, 0.1)[1])
    extended = s.extended(3.2)
    assert extended.theta_times[-1] == 3.2 and np.isfinite(extended.jet(3.05, 0.1)[1])
    np.testing.assert_allclose(
        extended.total_variance(2.5, 0.1), s.total_variance(2.5, 0.1), rtol=1e-14
    )
    slope = (s.theta_values[-1] - s.theta_values[-2]) / (TENORS[-1] - TENORS[-2])
    np.testing.assert_allclose(extended.theta(3.2), s.theta_values[-1] + 0.2 * slope, rtol=1e-13)


def test_reduces_to_ssvi_without_the_theta_dependence_and_at_short_times():
    # With rho_0 = rho_m the correlation is constant; with lambda = 1/2 and theta << 1 the factor
    # (1 + theta)^(lambda - 1) is close to one, so the slice is the SSVI slice.
    e = surface(rho_0=-0.6, rho_m=-0.6, lam=0.5)
    s = SSVI(e.theta_times, e.theta_values, -0.6, ETA, 0.5)
    theta = float(e.theta(0.25))
    k = np.linspace(-0.5, 0.5, 21)
    np.testing.assert_allclose(
        e.total_variance(0.25, k), s.total_variance(0.25, k) * 1.0, rtol=theta
    )
    np.testing.assert_allclose(
        e.initial_local_variance(0.0), s.initial_local_variance(0.0), rtol=1e-13
    )
    assert np.isnan(e.initial_local_variance(0.1))


def test_dupire_local_volatility_is_finite_and_positive_on_a_grid():
    local = Dupire(surface().extended(3.2))
    t = np.geomspace(1e-4, 3.1, 40)
    x = np.linspace(-0.8, 0.5, 27)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    sigma = np.asarray(local(tt, xx))
    assert np.isfinite(sigma).all() and (sigma > 0.05).all() and (sigma < 10.0).all()
    assert np.isfinite(local(0.0, 0.0)) and np.isnan(local(0.0, 0.1))
