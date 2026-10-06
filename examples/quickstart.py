"""Calibrate the leverage factor of a two-factor Bergomi model to a smile, by fixed-point iteration.

The quotes are made up from a known SSVI surface, so that the example runs without market data.
"""

import jax.numpy as jnp
import numpy as np

import locsvol  # noqa: F401  (enables 64-bit floating point)
from locsvol.backbones import Bergomi
from locsvol.calibration import fixed_point
from locsvol.leverage import (
    Chart,
    Clipped,
    Parametrized,
    quantile_bounds,
    quantile_envelope,
    table_times,
    tabulate,
)
from locsvol.pricing import implied_volatility, monte_carlo_prices
from locsvol.regression import ChebyshevTensor, GammaRegression
from locsvol.simulation import graded_times, simulate, with_dates
from locsvol.targets import SSVI, Dupire, fit_ssvi, fit_svi

# Quotes: implied volatilities at five maturities and 21 log strikes k = log(K/F).
truth = SSVI(jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 0.04]), rho=-0.6, eta=1.2, gamma=0.4)
maturities = jnp.repeat(jnp.asarray([0.1, 0.25, 0.5, 0.75, 1.0]), 21)
log_strikes = jnp.tile(jnp.linspace(-0.3, 0.2, 21), 5) * jnp.sqrt(maturities / 0.5)
quoted = truth.implied_volatility(maturities, log_strikes)
weights = jnp.full(maturities.size, 1.0 / maturities.size)

# Target: SSVI fit, refined slice by slice by raw SVI, and its Dupire local volatility.
ssvi = fit_ssvi(maturities, log_strikes, quoted, weights)
surface = fit_svi(maturities, log_strikes, quoted, weights, ssvi)
target = Dupire(surface)

# Coordinates of the leverage factor and the target, read at x clipped to the quantiles of the
# SSVI fit at 2e-4 and 1 - 2e-4 and tabulated; the state coordinate ends at the same quantiles.
horizon = 1.0
reference = ssvi.extended(horizon)
chart = Chart(
    reference.theta_times, reference.theta_values, reference.eta, reference.gamma, (0.0, horizon)
)
times = with_dates(graded_times(horizon, 0.25, 800, 2.0, 1 / 400), np.unique(maturities))
nodes = table_times(surface.expiries, horizon, 1e-6, 0.03, 40, 0.01, (0.0,), required=times)
local_volatility = tabulate(Clipped(target, *quantile_bounds(reference, nodes, 2e-4)), chart, nodes)
chart = chart.with_envelope(*quantile_envelope(chart, reference, nodes, 2e-4), time_clamp=times[1])

# Backbone and simulated variance factor.
backbone = Bergomi(nu=2.0, theta=0.2, rho1=-0.7, rho2=-0.3)
paths = simulate(backbone, times, dict(seed=1, role="calibration"), 16384)

# Fixed-point iteration with a Gamma regression on Chebyshev polynomials of degrees (6, 8).
basis = ChebyshevTensor((6, 8))
start = Parametrized(local_volatility, chart, basis, jnp.zeros(basis.size))
leverage = fixed_point(start, local_volatility, paths, GammaRegression(basis, ridge=1e-6), 5)[-1]

# Implied volatilities of the calibrated model on independent paths, against the quotes.
fresh = simulate(backbone, times, dict(seed=2, role="evaluation"), 65536)
calls = jnp.ones(maturities.size, dtype=bool)
prices = monte_carlo_prices(fresh, leverage, maturities, log_strikes, calls)
model = implied_volatility(prices, maturities, log_strikes, calls)
print(
    f"largest difference to the quoted implied volatilities: "
    f"{1e4 * float(jnp.max(jnp.abs(model - quoted))):.1f} bp"
)
