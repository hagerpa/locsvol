"""Fit of the power-law SSVI surface to implied volatilities (Gatheral and Jacquier,
Quant. Finance 14, 2014).

Given quotes with maturities T_j, log-moneyness k_j, implied volatilities sigma_j and weights
omega_j > 0 at the expiries T_1 < ... < T_n, the fit minimizes the weighted least-squares error in
implied volatility

    F = 1/2 sum_j omega_j / (sum_l omega_l) (sqrt(w(T_j, k_j) / T_j) - sigma_j)^2

over theta_1 < ... < theta_n, rho, eta and gamma, subject to Gatheral and Jacquier's conditions
for the absence of butterfly arbitrage with a margin,

    4 - theta_n phi (1 +- rho) >= margin,    4 - theta_n phi^2 (1 +- rho) >= margin.

The ATM total variance is parametrized by its forward variances xi_i = (theta_i - theta_{i-1}) /
(T_i - T_{i-1}) > 0, so that it increases, and the fitted vector is
(log xi_1, ..., log xi_n, rho, log eta, gamma) within bounds. The fit starts from theta_i, the
weighted mean of T sigma^2 over the quotes of T_i nearest the money, raised where needed so that
every xi_i is at least its lower bound, and from given rho, eta and gamma. It solves the problem
first without the conditions by the trust-region reflective method of SciPy's least_squares and
then with them by SLSQP (Kraft, DFVLR-FB 88-28, 1988) from that point.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

from locsvol.errors import ConvergenceError
from locsvol.targets.ssvi import SSVI, linear_interpolation, ssvi_total_variance

LEAST_SQUARES = dict(method="trf", loss="linear", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
SLSQP = dict(ftol=1e-12, maxiter=1000)
FEASIBILITY = 1e-10  # admissible violation of a condition by the SLSQP solution


def _surface(parameters, expiries, log_moneyness_bounds=(-20.0, 20.0)):
    """The SSVI surface of the vector (log xi_1, ..., log xi_n, rho, log eta, gamma)."""
    n = expiries.size
    times = jnp.concatenate((jnp.zeros(1), expiries))
    theta = jnp.cumsum(jnp.exp(parameters[:n]) * jnp.diff(times))
    return SSVI(
        times,
        jnp.concatenate((jnp.zeros(1), theta)),
        parameters[n],
        jnp.exp(parameters[n + 1]),
        parameters[n + 2],
        log_moneyness_bounds,
    )


@jax.jit
def _residuals(parameters, quotes, expiries):
    t, k, sigma, weights = quotes
    s = _surface(parameters, expiries)
    theta, _ = linear_interpolation(s.theta_times, s.theta_values, t)
    model = jnp.sqrt(ssvi_total_variance(theta, k, s.rho, s.eta, s.gamma) / t)
    return jnp.sqrt(weights / jnp.sum(weights)) * (model - sigma)


@jax.jit
def _objective(parameters, quotes, expiries):
    residuals = _residuals(parameters, quotes, expiries)
    return 0.5 * jnp.vdot(residuals, residuals)


@jax.jit
def _conditions(parameters, expiries, margins):
    return _surface(parameters, expiries).conditions() - margins


_jacobian = jax.jit(jax.jacfwd(_residuals))
_gradient = jax.jit(jax.grad(_objective))
_conditions_jacobian = jax.jit(jax.jacfwd(_conditions))


def _log_bounds(lower, upper):
    """Bounds on log x whose exponentials lie in [lower, upper]."""
    low, high = math.log(lower), math.log(upper)
    while float(jnp.exp(low)) < lower:
        low = np.nextafter(low, np.inf)
    while float(jnp.exp(high)) > upper:
        high = np.nextafter(high, -np.inf)
    return low, high


def _starting_theta(maturities, log_moneyness, implied_volatilities, weights, expiries, slope):
    """theta_i = max(theta_i^ATM, theta_{i-1} + slope (T_i - T_{i-1})), with theta_i^ATM the
    weighted mean of T sigma^2 over the quotes of T_i nearest the money."""
    theta, previous, start = [], 0.0, 0.0
    for expiry in expiries:
        at = maturities == expiry
        distance = np.abs(log_moneyness[at])
        nearest = distance == distance.min()
        w = weights[at][nearest]
        observed = np.sum(w * expiry * implied_volatilities[at][nearest] ** 2) / np.sum(w)
        previous = max(observed, previous + slope * (expiry - start))
        theta.append(previous)
        start = expiry
    return np.asarray(theta)


def fit_ssvi(
    maturities,
    log_moneyness,
    implied_volatilities,
    weights,
    *,
    forward_variance_bounds=(1e-6, 4.0),
    rho_bounds=(-0.999999, 0.999999),
    eta_bounds=(1e-6, 4.0),
    gamma_bounds=(0.01, 0.5),
    start=(-0.4, 0.2, 0.4),
    margins=(1e-6, 1e-6),
    log_moneyness_bounds=(-20.0, 20.0),
):
    """The SSVI surface fitted to the quotes; see the module docstring.

    start holds (rho, eta, gamma) of the starting point and margins the margins of the linear and
    the quadratic conditions. Quotes with zero weight are ignored; every expiry needs a quote of
    positive weight.
    """
    t, k, sigma, w = (
        np.asarray(v, dtype=float)
        for v in (maturities, log_moneyness, implied_volatilities, weights)
    )
    active = w > 0.0
    t, k, sigma, w = t[active], k[active], sigma[active], w[active]
    expiries = np.unique(t)
    if not np.all(np.isin(np.unique(np.asarray(maturities, dtype=float)), expiries)):
        raise ValueError("every expiry needs a quote of positive weight")

    times = np.concatenate(([0.0], expiries))
    theta = _starting_theta(t, k, sigma, w, expiries, forward_variance_bounds[0])
    slopes = np.maximum(
        np.diff(np.concatenate(([0.0], theta))) / np.diff(times), forward_variance_bounds[0]
    )
    rho, eta, gamma = start
    x0 = np.concatenate((np.log(slopes), [rho, np.log(eta), gamma]))
    slope_low, slope_high = _log_bounds(*forward_variance_bounds)
    eta_low, eta_high = _log_bounds(*eta_bounds)
    n = expiries.size
    lower = np.concatenate((np.full(n, slope_low), [rho_bounds[0], eta_low, gamma_bounds[0]]))
    upper = np.concatenate((np.full(n, slope_high), [rho_bounds[1], eta_high, gamma_bounds[1]]))
    x0 = np.clip(x0, lower, upper)

    quotes = tuple(jnp.asarray(v) for v in (t, k, sigma, w))
    expiries = jnp.asarray(expiries)
    margins = jnp.asarray([margins[0], margins[0], margins[1], margins[1]])
    unconditioned = least_squares(
        lambda x: np.asarray(_residuals(x, quotes, expiries)),
        x0,
        jac=lambda x: np.asarray(_jacobian(x, quotes, expiries)),
        bounds=(lower, upper),
        **LEAST_SQUARES,
    )
    result = minimize(
        lambda x: float(_objective(x, quotes, expiries)),
        unconditioned.x,
        jac=lambda x: np.asarray(_gradient(x, quotes, expiries)),
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=dict(
            type="ineq",
            fun=lambda x: np.asarray(_conditions(x, expiries, margins)),
            jac=lambda x: np.asarray(_conditions_jacobian(x, expiries, margins)),
        ),
        options=SLSQP,
    )
    if not result.success:
        raise ConvergenceError(f"SSVI fit: {result.message}")
    if np.min(_conditions(result.x, expiries, margins)) < -FEASIBILITY:
        raise ConvergenceError("SSVI fit violates the conditions of Gatheral and Jacquier")
    return _surface(jnp.asarray(result.x), expiries, log_moneyness_bounds)
