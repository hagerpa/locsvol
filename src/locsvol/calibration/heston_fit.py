"""The Heston backbone fitted to the quotes of an earlier day.

In the Heston model (Heston, Rev. Financ. Stud. 6, 1993)

    dS_t / S_t = sqrt(V_t) dW_t,   dV_t = kappa (theta - V_t) dt + sigma sqrt(V_t) dZ_t,
    d<W, Z>_t = rho dt,   V_0 = v_0,

call prices are Fourier integrals of the characteristic function of log S_T, which is known in
closed form. They are computed on Lewis' contour (Lewis, Option Valuation under Stochastic
Volatility, 2000) by the package roughprix, which prices the out-of-the-money option of each
strike and the call by put-call parity. The parameters minimize the vega-scaled distance of the
model call prices P_q to the midpoints m_q of the quotes,

    sum_q w_q (P_q - m_q)^2 / v_q^2,

by the trust region reflective method of least squares within box bounds (Branch, Coleman and
Li, SIAM J. Sci. Comput. 21, 1999), with the Jacobian by forward differences.
"""

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from locsvol.backbones.heston import Heston
from locsvol.calibration.gauss_newton import interval_residuals
from locsvol.errors import ConvergenceError

START = (2.0, 0.04, 1.0, -0.7, 0.03)  # kappa, theta, sigma, rho, v0
BOUNDS = ((0.01, 1e-4, 0.01, -0.999, 1e-4), (50.0, 1.0, 10.0, 0.999, 1.0))
FOURIER = dict(S_0=1.0, call=True, payoff_selection="otm", R=0.5, integration="panel")
SOLVER = dict(
    method="trf", jac="2-point", x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=500
)


def heston_call_prices(parameters, maturities, log_strikes):
    """Call prices divided by discount factor and forward for (kappa, theta, sigma, rho, v0).

    Requires the package roughprix (the optional dependency heston).
    """
    from roughprix import RoughHeston, RoughHestonParam

    kappa, theta, sigma, rho, v0 = parameters
    # roughprix takes the drift kappa (theta - V) as theta' - lam V with theta' = kappa theta.
    model = RoughHeston(
        RoughHestonParam(lam=kappa, theta=kappa * theta, nu=sigma, rho=rho, V0=v0, H=0.5), 0.0
    )
    maturities = np.asarray(maturities)
    strikes = np.exp(np.asarray(log_strikes))
    prices = np.empty(maturities.size)
    for maturity in np.unique(maturities):
        rows = maturities == maturity
        prices[rows] = model.eur_call_put(strikes[rows], float(maturity), **FOURIER)
    return prices


def fit_heston(
    maturities, log_strikes, midpoints, weights, vegas, start=START, bounds=BOUNDS, floor=1e-8
):
    """The Heston backbone whose call prices fit the midpoints; see the module docstring."""

    def residuals(parameters):
        prices = heston_call_prices(parameters, maturities, log_strikes)
        return np.asarray(
            interval_residuals(jnp.asarray(prices), midpoints, midpoints, weights, vegas)
        )

    result = least_squares(residuals, np.asarray(start), bounds=bounds, **SOLVER)
    if not result.success:
        raise ConvergenceError(f"Heston least squares: {result.message}")
    kappa, theta, sigma, rho, v0 = (float(value) for value in result.x)
    return Heston(kappa, theta, sigma, rho, v0, floor)
