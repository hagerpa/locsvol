"""The SSVI and refined SVI fits and the short extension, on quotes of a known SSVI surface."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import locsvol  # noqa: F401
from locsvol.targets import SSVI, Dupire, RefinedSVI, fit_ssvi, fit_svi, short_extension
from locsvol.targets.dupire import butterfly_factor
from locsvol.targets.log_contract import log_contract_variance
from locsvol.targets.refined_fit import constraint_grid
from locsvol.targets.svi import svi, svi_derivatives, svi_from_ssvi

EXPIRIES = np.asarray([0.08, 0.25, 0.6])
THETA = np.asarray([0.0035, 0.0105, 0.024])
RHO, ETA, GAMMA = -0.45, 0.9, 0.35


@pytest.fixture(scope="module")
def quotes():
    surface = SSVI(
        jnp.asarray(np.concatenate(([0.0], EXPIRIES))),
        jnp.asarray(np.concatenate(([0.0], THETA))),
        RHO,
        ETA,
        GAMMA,
    )
    t = np.repeat(EXPIRIES, 21)
    k = np.concatenate([np.linspace(-0.3, 0.2, 21) * np.sqrt(e / 0.25) for e in EXPIRIES])
    sigma = np.asarray(surface.implied_volatility(t, k))
    weights = np.where(k < 0, 2.0, 1.0)
    return t, k, sigma, weights


@pytest.fixture(scope="module")
def ssvi(quotes):
    return fit_ssvi(*quotes)


def test_ssvi_fit_recovers_the_surface(ssvi):
    np.testing.assert_allclose(ssvi.theta_values[1:], THETA, rtol=1e-7)
    np.testing.assert_allclose([ssvi.rho, ssvi.eta, ssvi.gamma], [RHO, ETA, GAMMA], rtol=1e-6)
    assert np.all(np.asarray(ssvi.conditions()) >= 1e-6 - 1e-10)


def test_ssvi_fit_keeps_the_conditions_with_their_margin():
    # Quotes of an SSVI with theta phi^2 (1 + |rho|) > 4 at the last expiry.
    surface = SSVI(jnp.asarray([0.0, 0.5, 1.0]), jnp.asarray([0.0, 0.02, 0.05]), -0.7, 3.0, 0.5)
    t = np.repeat([0.5, 1.0], 15)
    k = np.tile(np.linspace(-0.4, 0.3, 15), 2)
    fitted = fit_ssvi(
        t, k, surface.implied_volatility(t, k), np.ones(t.size), eta_bounds=(1e-6, 4.0)
    )
    assert np.min(np.asarray(fitted.conditions())) >= 1e-6 - 1e-10
    assert np.min(np.asarray(surface.conditions())) < 0


def test_ssvi_fit_needs_a_quote_at_every_expiry(quotes):
    t, k, sigma, weights = quotes
    with pytest.raises(ValueError):
        fit_ssvi(t, k, sigma, np.where(t == EXPIRIES[1], 0.0, weights))


def test_refined_fit_matches_the_quotes_without_arbitrage(quotes, ssvi):
    t, k, sigma, weights = quotes
    surface = fit_svi(t, k, sigma, weights, ssvi)
    fitted = np.asarray(surface.implied_volatility(t, k))
    assert np.sqrt(np.average((fitted - sigma) ** 2, weights=weights)) < 1e-5
    grid = jnp.asarray(constraint_grid(k))
    values = jax.vmap(lambda p: svi(p, grid))(surface.slices)
    slopes, curvatures = jax.vmap(lambda p: svi_derivatives(p, grid))(surface.slices)
    assert np.all(np.diff(np.asarray(values), axis=0) >= 1e-6 - 1e-10)
    assert np.all(np.asarray(butterfly_factor(values, grid, slopes, curvatures)) >= 0.03 - 1e-10)
    # The short extension makes the log-contract forward variance continuous at T_1.
    first, step = float(EXPIRIES[0]), 1e-6
    v = log_contract_variance(surface, jnp.asarray([first - step, first, first + step]))
    below, above = (v[1] - v[0]) / step, (v[2] - v[1]) / step
    assert abs(float(below - above)) < 1e-9
    assert np.all(
        np.isfinite(Dupire(surface)(jnp.asarray([0.0, 0.01, 0.3])[:, None], k[None, :21]))
    )


def refined(first_theta):
    theta = np.asarray([first_theta, 0.0105, 0.024])
    phi = ETA * theta**-GAMMA
    slices = jnp.stack([svi_from_ssvi(w, RHO, p) for w, p in zip(theta, phi, strict=True)])
    return RefinedSVI(jnp.asarray(EXPIRIES), slices)


def jump(surface):
    first, step = float(surface.expiries[0]), 1e-6
    v = log_contract_variance(surface, jnp.asarray([first - step, first, first + step]))
    return float((v[1] - v[0]) / step - (v[2] - v[1]) / step)


@pytest.mark.parametrize("first_theta", [0.0035, 0.004, 0.005])
def test_short_extension_makes_the_forward_variance_continuous(first_theta):
    surface = short_extension(refined(first_theta))
    t0, c, first = float(surface.shift), float(surface.level), float(EXPIRIES[0])
    assert abs(jump(surface)) < 1e-9
    if first_theta == 0.004:  # the shift alone suffices
        assert c == 0.0 and 0.0 < t0 < 1.0
    elif first_theta == 0.005:  # front-loaded, on the butterfly bound
        assert c > 0.0 and c == t0 / (first + t0)
    else:  # back-loaded, the shift at the end of its bracket
        assert t0 == 1.0 and -1.0 <= c < 0.0


def test_short_extension_without_a_solution():
    with pytest.raises(ValueError):
        short_extension(refined(0.005), with_level=False)
    theta = np.asarray([0.002, 0.014, 0.03])
    phi = ETA * theta**-GAMMA
    slices = jnp.stack([svi_from_ssvi(w, RHO, p) for w, p in zip(theta, phi, strict=True)])
    with pytest.raises(ValueError):
        short_extension(RefinedSVI(jnp.asarray(EXPIRIES), slices))
