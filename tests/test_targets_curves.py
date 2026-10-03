"""Log-contract and forward variance, digital quantiles, quote calculations, the analytic targets
and the Heston projection."""

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

import locsvol  # noqa: F401
from locsvol.backbones import Heston
from locsvol.leverage import Chart
from locsvol.regression import ChebyshevTensor, GammaRegression
from locsvol.targets import (
    ConstantVolatility,
    RefinedSVI,
    Synthetic,
    TermStructure,
    digital_probability,
    digital_quantiles,
    forward_variance,
    heston_projection,
    log_contract_variance,
    log_moneyness,
    lognormal_quantiles,
    otm_implied_volatility,
    put_call_parity,
)
from locsvol.targets.log_contract import log_contract_quadrature
from locsvol.targets.svi import svi_from_ssvi


def test_quadrature_weights():
    nodes, weights = log_contract_quadrature(64, -3.0, 2.0)
    assert nodes.shape == weights.shape == (128,)
    assert np.all(
        (nodes[:64] > -3.0) & (nodes[:64] < 0.0) & (nodes[64:] > 0.0) & (nodes[64:] < 2.0)
    )
    np.testing.assert_allclose(jnp.sum(weights * jnp.exp(nodes) / 2), 5.0, rtol=1e-14)


def test_log_contract_of_flat_smiles():
    flat = ConstantVolatility(0.2)
    t = jnp.asarray([0.0, 0.02, 0.5, 1.0])
    np.testing.assert_allclose(
        log_contract_variance(flat, t, bounds=(-20.0, 20.0)), 0.04 * t, rtol=1e-11
    )
    times = jnp.asarray([0.0, 0.1, 0.3, 0.7, 1.0])
    structure = TermStructure(times, jnp.asarray([0.01, 0.04, 0.02, 0.09]))
    np.testing.assert_allclose(
        forward_variance(structure, times, bounds=(-20.0, 20.0)),
        structure.forward_variances,
        rtol=1e-11,
    )


def test_digital_probabilities_and_quantiles_of_a_lognormal_law():
    flat = ConstantVolatility(0.25)
    t, k = jnp.asarray([0.1, 1.0])[:, None], jnp.linspace(-1.0, 0.5, 7)[None, :]
    w = 0.0625 * t
    np.testing.assert_allclose(
        digital_probability(flat, t, k), ndtr((k + w / 2) / jnp.sqrt(w)), rtol=1e-13
    )
    p = jnp.asarray([0.001, 0.3, 0.999])
    brackets = jnp.asarray([[-5.0, 5.0], [-5.0, 5.0]])
    q = digital_quantiles(flat, t[:, 0], p, brackets)
    np.testing.assert_allclose(q, jnp.log(lognormal_quantiles(w[:, 0], p)), atol=1e-9)
    narrow = digital_quantiles(flat, t[:, 0], p, jnp.asarray([[-0.1, 0.1], [-0.1, 0.1]]))
    assert np.isnan(narrow[0, 0]) and np.isfinite(narrow[0, 1])


def test_digital_quantiles_of_a_smile():
    theta, rho, eta, gamma = np.asarray([0.01, 0.03]), -0.6, 1.2, 0.4
    slices = jnp.stack([svi_from_ssvi(w, rho, eta * w**-gamma) for w in theta])
    surface = RefinedSVI(jnp.asarray([0.25, 1.0]), slices, shift=0.1)
    times, p = jnp.asarray([0.1, 0.5]), jnp.asarray([0.001, 0.5, 0.999])
    q = digital_quantiles(surface, times, p, jnp.tile(jnp.asarray([-20.0, 20.0]), (2, 1)))
    np.testing.assert_allclose(
        digital_probability(surface, times[:, None], q), jnp.broadcast_to(p, q.shape), atol=1e-10
    )
    assert np.all(np.diff(np.asarray(q), axis=1) > 0)


def test_put_call_parity_and_quote_coordinates():
    strikes = jnp.asarray([80.0, 90.0, 100.0, 110.0, 120.0, 1.0])
    discount, forward = 0.97, 101.5
    calls = jnp.maximum(forward - strikes, 0.0) * discount + 3.0
    puts = calls - discount * (forward - strikes)
    weights = jnp.asarray([1.0, 2.0, 1.0, 1.0, 0.5, 0.0])  # the last quote is padding
    puts = puts.at[-1].set(1e6)
    d, f = put_call_parity(strikes, calls, puts, weights)
    np.testing.assert_allclose([d, f], [discount, forward], rtol=1e-13)
    k = log_moneyness(strikes[:5], forward)
    np.testing.assert_allclose(k, np.log(np.asarray(strikes[:5]) / forward), rtol=1e-15)
    iv = otm_implied_volatility(k, jnp.full(5, 0.2), jnp.full(5, 0.3))
    np.testing.assert_array_equal(iv, np.where(np.asarray(k) >= 0, 0.2, 0.3))


def test_analytic_targets():
    flat = ConstantVolatility(0.3)
    np.testing.assert_array_equal(flat(jnp.asarray([0.0, 1.0]), 0.5), [0.3, 0.3])
    assert np.isnan(flat(-1.0, 0.0))
    times = jnp.asarray([0.0, 0.5, 1.0])
    structure = TermStructure(times, jnp.asarray([0.04, 0.09]))
    np.testing.assert_allclose(
        structure(jnp.asarray([0.0, 0.49, 0.5, 1.0]), 0.0), [0.2, 0.2, 0.3, 0.3]
    )
    np.testing.assert_allclose(
        structure.total_variance(jnp.asarray([0.5, 0.75, 1.0]), 0.1), [0.02, 0.0425, 0.065]
    )
    assert np.isnan(structure(1.01, 0.0))
    synthetic = Synthetic(
        jnp.asarray([0.15, 0.3, 0.6]),
        jnp.asarray([0.5, 0.3, 0.2]),
        jnp.asarray([0.1, 0.1]),
        jnp.asarray([0.5, 0.5]),
        jnp.asarray([1.0, 0.5]),
        1.0,
    )
    sigmas, p = np.asarray([0.15, 0.3, 0.6]), np.asarray([0.5, 0.3, 0.2])
    origin = np.sqrt(0.25 * min(abs(np.sum(p * sigmas) / (np.sum(p / sigmas) + 0.01)), 2.0))
    np.testing.assert_allclose(synthetic(0.0, 0.0), origin, rtol=1e-15)
    values = synthetic(jnp.asarray([0.05, 0.5])[:, None], jnp.linspace(-2.0, 2.0, 9)[None, :])
    assert np.all(np.isfinite(values) & (np.asarray(values) > 0))
    assert np.isnan(synthetic(0.0, 0.1))


def test_heston_projection_of_a_nearly_deterministic_variance():
    heston = Heston(kappa=1.5, theta=0.04, sigma=0.01, rho=-0.5, v0=0.0225)
    times = jnp.linspace(0.0, 1.0, 21)
    chart = Chart(
        jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 0.04]), 1.0, 0.5, time_bounds=(0.0, 1.0)
    )
    regression = GammaRegression(ChebyshevTensor((3, 2)), ridge=1e-8)
    projection = heston_projection(heston, times, dict(seed=7), 1024, chart, regression)
    t = jnp.asarray([0.2, 0.5, 0.9])
    mean = 0.04 + (0.0225 - 0.04) * jnp.exp(-1.5 * t)
    np.testing.assert_allclose(projection(t, 0.0), jnp.sqrt(mean), rtol=5e-3)
