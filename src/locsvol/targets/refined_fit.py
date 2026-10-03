"""Fit of the refined SVI surface: a raw SVI slice per expiry by weighted least squares in implied
volatility, under constraints against static arbitrage.

The slices start from those of a fitted SSVI surface (svi.svi_from_ssvi) and are fitted one after
another, each with its neighbours fixed, in sweeps over all expiries. A sweep is the last when it
changes neither the slices' total variances on the constraint grid nor the objective
sum_j omega_j / (sum_l omega_l) (sqrt(w(T_j, k_j) / T_j) - sigma_j)^2 / 2 of all quotes by more
than a relative tolerance. The slice p = (a, b, rho, m, sigma) at expiry T minimizes

    1/2 sum_j omega_j / (sum_l omega_l) (sqrt(w_p(k_j) / T) - sigma_j)^2

over the quotes of T, within bounds on p, by SLSQP (Kraft, DFVLR-FB 88-28, 1988) subject to

    a + b sigma sqrt(1 - rho^2) >= w_min                   positive total variance,
    b (1 - rho) <= s_max,  b (1 + rho) <= s_max            wing slopes below 2 (Lee, Math. Finance
                                                           14, 2004),
    g(k) >= g_min,  w_p(k) >= w_min                        no butterfly arbitrage,
    w_previous(k) + delta <= w_p(k) <= w_next(k) - delta   no calendar arbitrage,

at the log-moneyness k of a grid, with g the density factor of dupire.butterfly_factor. When
SLSQP fails from a start that meets the bounds and constraints, the slice is fitted again from its
SSVI slice and that fit is taken if it succeeds with an objective no larger than the start's;
otherwise the slice keeps its start. The fitted surface's short extension is then fixed by
refined.short_extension.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from locsvol.errors import ConvergenceError
from locsvol.targets.dupire import butterfly_factor
from locsvol.targets.refined import RefinedSVI, short_extension
from locsvol.targets.svi import (
    svi,
    svi_derivatives,
    svi_from_ssvi,
    svi_minimum,
    svi_valid,
    svi_wings,
)

SLSQP = dict(ftol=1e-12, maxiter=1000)
FEASIBILITY = 1e-10  # admissible violation of a constraint by an SLSQP solution
LOWER = (-10.0, 1e-8, -0.999999, -20.0, 1e-6)  # bounds on (a, b, rho, m, sigma)
UPPER = (10.0, 2.0, 0.999999, 20.0, 20.0)


def constraint_grid(log_moneyness):
    """1201 equally spaced points from 0.5 below to 0.5 above the quotes' log-moneyness, 120
    geometrically spaced points on to -20 and to 20, and 0."""
    left, right = float(np.min(log_moneyness)) - 0.5, float(np.max(log_moneyness)) + 0.5
    grid = set(np.linspace(left, right, 1201))
    grid |= set(-np.geomspace(-left, 20.0, 120)) | set(np.geomspace(right, 20.0, 120)) | {0.0}
    return np.asarray(sorted(grid))


@jax.jit
def _cost(parameters, t, k, sigma, weights):
    weights = weights / jnp.max(weights)
    weights = weights / jnp.sum(weights)
    w = jnp.where(svi_valid(parameters), svi(parameters, k), jnp.nan)
    residuals = jnp.where(weights == 0.0, 0.0, jnp.sqrt(weights) * (jnp.sqrt(w / t) - sigma))
    return 0.5 * jnp.sum(residuals**2)


@jax.jit
def _slacks(parameters, previous, following, grid, limits):
    """The constraints of the module docstring as slacks, nonnegative where they hold."""
    minimum_variance, maximum_slope, minimum_factor, margin = limits
    w = svi(parameters, grid)
    w_k, w_kk = svi_derivatives(parameters, grid)
    left, right = svi_wings(parameters)
    slacks = [
        jnp.atleast_1d(svi_minimum(parameters) - minimum_variance),
        jnp.stack((maximum_slope - left, maximum_slope - right)),
        butterfly_factor(w, grid, w_k, w_kk) - minimum_factor,
        w - minimum_variance,
    ]
    if previous is not None:
        slacks.append(w - svi(previous, grid) - margin)
    if following is not None:
        slacks.append(svi(following, grid) - w - margin)
    return jnp.concatenate(slacks)


_gradient = jax.jit(jax.grad(_cost))
_slacks_jacobian = jax.jit(jax.jacfwd(_slacks))


def fit_slice(quotes, start, previous, following, grid, limits, bounds):
    """The SLSQP fit of one slice from the start, and whether SLSQP succeeded."""
    lower, upper = bounds
    result = minimize(
        lambda x: float(_cost(x, *quotes)),
        np.asarray(start),
        jac=lambda x: np.asarray(_gradient(x, *quotes)),
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=dict(
            type="ineq",
            fun=lambda x: np.asarray(_slacks(x, previous, following, grid, limits)),
            jac=lambda x: np.asarray(_slacks_jacobian(x, previous, following, grid, limits)),
        ),
        options=SLSQP,
    )
    return jnp.asarray(result.x), result.success


def _feasible(parameters, quotes, previous, following, grid, limits, bounds):
    """Whether a slice meets its bounds and constraints and has a finite objective."""
    lower, upper = bounds
    x = np.asarray(parameters)
    slacks = np.asarray(_slacks(parameters, previous, following, grid, limits))
    return bool(
        np.all((x >= lower) & (x <= upper))
        and np.isfinite(float(_cost(parameters, *quotes)))
        and np.all(slacks >= -FEASIBILITY)
        and svi_valid(parameters)
    )


@jax.jit
def _objective(slices, quotes, expiries):
    t, k, sigma, weights = quotes
    index = jnp.searchsorted(expiries, t)
    w = jax.vmap(lambda p, x: jnp.where(svi_valid(p), svi(p, x), jnp.nan))(slices[index], k)
    weights = weights / jnp.max(weights)
    weights = weights / jnp.sum(weights)
    return 0.5 * jnp.sum(weights * (jnp.sqrt(w / t) - sigma) ** 2)


@jax.jit
def _grid_values(slices, grid):
    return jax.vmap(lambda p: jnp.where(svi_valid(p), svi(p, grid), jnp.nan))(slices)


def fit_svi(
    maturities,
    log_moneyness,
    implied_volatilities,
    weights,
    ssvi,
    *,
    grid=None,
    bounds=(LOWER, UPPER),
    minimum_variance=1e-8,
    maximum_slope=1.999999,
    minimum_factor=0.03,
    margin=1e-6,
    sweeps=40,
    tolerance=1e-4,
    long_forward_variance=0.04,
    maximum_maturity=1.2,
    log_moneyness_bounds=(-20.0, 20.0),
    resolution=26.0,
    extension=None,
):
    """The refined SVI surface fitted to the quotes from the SSVI surface ssvi.

    grid defaults to constraint_grid of the quotes' log-moneyness; minimum_variance, maximum_slope,
    minimum_factor and margin are w_min, s_max, g_min and delta of the module docstring.
    long_forward_variance, maximum_maturity, log_moneyness_bounds and resolution define the surface
    (refined.RefinedSVI), and extension holds keyword arguments of refined.short_extension.
    """
    t, k, sigma, w = (
        np.asarray(v, dtype=float)
        for v in (maturities, log_moneyness, implied_volatilities, weights)
    )
    expiries = np.unique(t)
    grid = jnp.asarray(constraint_grid(k) if grid is None else grid)
    limits = (minimum_variance, maximum_slope, minimum_factor, margin)
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    theta = ssvi.theta(expiries)
    phi = ssvi.eta * theta**-ssvi.gamma
    seeds = [svi_from_ssvi(theta[i], ssvi.rho, phi[i]) for i in range(expiries.size)]
    slices = list(seeds)
    all_quotes = tuple(jnp.asarray(v) for v in (t, k, sigma, w))
    quotes = [
        (jnp.asarray(expiry), *(jnp.asarray(v[t == expiry]) for v in (k, sigma, w)))
        for expiry in expiries
    ]

    for _ in range(sweeps):
        before = jnp.stack(slices)
        cost = float(_objective(before, all_quotes, jnp.asarray(expiries)))
        for i in range(expiries.size):
            previous = slices[i - 1] if i > 0 else None
            following = slices[i + 1] if i + 1 < expiries.size else None
            arguments = (previous, following, grid, limits, (lower, upper))
            fitted, solved = fit_slice(quotes[i], slices[i], *arguments)
            if not solved:
                fitted = _recover(quotes[i], slices[i], seeds[i], *arguments)
            elif not _feasible(fitted, quotes[i], *arguments):
                raise ConvergenceError("refined SVI: a slice violates its constraints")
            slices[i] = fitted
        after = jnp.stack(slices)
        w_before, w_after = _grid_values(before, grid), _grid_values(after, grid)
        change = float(jnp.max(jnp.abs(w_after - w_before) / w_before))
        new_cost = float(_objective(after, all_quotes, jnp.asarray(expiries)))
        if not (np.isfinite(change) and np.isfinite(new_cost)):
            raise ConvergenceError("refined SVI: the slices are not finite")
        if change <= tolerance and abs(new_cost - cost) / max(1.0, abs(cost)) <= tolerance:
            break
    else:
        raise ConvergenceError(f"refined SVI: no convergence in {sweeps} sweeps")

    surface = RefinedSVI(
        jnp.asarray(expiries),
        jnp.stack(slices),
        long_forward_variance=long_forward_variance,
        maximum_maturity=maximum_maturity,
        log_moneyness_bounds=log_moneyness_bounds,
        resolution=resolution,
    )
    if not bool(jnp.all(jnp.diff(surface.atm()) > 0.0)):
        raise ConvergenceError("refined SVI: the ATM total variance does not increase")
    return short_extension(surface, **(extension or {}))


def _recover(quotes, start, seed, *arguments):
    """The slice after SLSQP failed from a feasible start: the fit from the SSVI slice if it is
    feasible and no worse than the start, otherwise the start."""
    if not _feasible(start, quotes, *arguments):
        raise ConvergenceError("refined SVI: SLSQP failed from an infeasible start")
    if not np.array_equal(np.asarray(start), np.asarray(seed)):
        fitted, solved = fit_slice(quotes, seed, *arguments)
        if (
            solved
            and _feasible(fitted, quotes, *arguments)
            and float(_cost(fitted, *quotes)) <= float(_cost(start, *quotes))
        ):
            return fitted
    return start
