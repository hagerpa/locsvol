import jax.numpy as jnp
import numpy as np

import locsvol  # noqa: F401
from locsvol.backbones.bergomi import ForwardVariance
from locsvol.backbones.heston import Heston
from locsvol.calibration.heston_fit import fit_heston, heston_call_prices
from locsvol.leverage.deterministic import Deterministic
from locsvol.pricing.black import black_price, black_vega, implied_volatility
from locsvol.pricing.monte_carlo import monte_carlo_prices
from locsvol.simulation.paths import simulate

MATURITIES = np.repeat([0.25, 1.0], 5)
LOG_STRIKES = np.tile(np.linspace(-0.25, 0.15, 5), 2)


def test_small_volatility_of_variance_gives_black_prices():
    prices = heston_call_prices((2.0, 0.04, 0.01, -0.5, 0.04), MATURITIES, LOG_STRIKES)
    black = black_price(0.04 * MATURITIES, LOG_STRIKES, True)
    np.testing.assert_allclose(prices, black, atol=5e-4)


def test_fit_recovers_the_parameters_of_its_own_prices():
    truth = (1.5, 0.05, 0.6, -0.6, 0.03)
    midpoints = heston_call_prices(truth, MATURITIES, LOG_STRIKES)
    vegas = black_vega(
        implied_volatility(midpoints, MATURITIES, LOG_STRIKES, True), MATURITIES, LOG_STRIKES
    )
    fitted = fit_heston(
        MATURITIES,
        LOG_STRIKES,
        midpoints,
        np.full(10, 0.1),
        vegas,
        start=(1.6, 0.048, 0.65, -0.55, 0.032),
    )
    found = (fitted.kappa, fitted.theta, fitted.sigma, fitted.rho, fitted.v0)
    np.testing.assert_allclose(found, truth, rtol=1e-6)


def test_leverage_of_the_mean_variance_gives_the_heston_prices():
    parameters = (2.0, 0.04, 0.6, -0.7, 0.03)
    model = Heston(*parameters)
    times = jnp.linspace(0.0, 1.0, 201)
    paths = simulate(model, times, {"seed": 3, "replication": 0}, 2**14)
    leverage = Deterministic(ForwardVariance(times[1:], model.mean(times)))
    calls = np.ones(MATURITIES.size, dtype=bool)
    prices = monte_carlo_prices(
        paths, leverage, MATURITIES, LOG_STRIKES, calls, out_of_the_money=True
    )
    exact = heston_call_prices(parameters, MATURITIES, LOG_STRIKES)
    np.testing.assert_allclose(
        implied_volatility(prices, MATURITIES, LOG_STRIKES, calls),
        implied_volatility(exact, MATURITIES, LOG_STRIKES, calls),
        atol=5e-3,
    )
