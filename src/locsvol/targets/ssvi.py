"""The power-law SSVI surface (Gatheral and Jacquier, Arbitrage-free SVI volatility surfaces,
Quant. Finance 14, 2014).

The total variance at maturity T and log-moneyness k = log(K/F) is

    w(T, k) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)),   phi = eta theta^-gamma,

where theta = theta(T) = w(T, 0) is the ATM total variance, linear between knots
(0, 0), (T_1, theta_1), ..., (T_n, theta_n) with increasing theta_i. With z = theta phi k and
R = sqrt((z + rho theta)^2 + (1 - rho^2) theta^2), w = (theta + rho z)/2 + R/2; where
theta + rho z < 0 the two halves cancel, and the equal w = (1 - rho^2) z^2 / (2 (R - theta - rho z))
is evaluated instead. The derivatives are

    w_k = theta phi B / 2,    w_kk = theta phi^2 (1 - rho^2) theta^3 / (2 R^3),
    w_T = theta'(T) (w - gamma z B / 2) / theta,    B = rho + (z + rho theta) / R,

with the right derivative theta'(T) at the knots. With increasing theta and 0 < gamma < 1 the
surface is free of calendar arbitrage (Gatheral and Jacquier, Theorem 4.1). For gamma <= 1/2,
theta phi and theta phi^2 increase with theta, and the surface is free of butterfly arbitrage if
theta phi (1 + |rho|) < 4 and theta phi^2 (1 + |rho|) <= 4 at the largest theta (Theorem 4.2).
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


def ssvi_total_variance(theta, log_moneyness, rho, eta, gamma):
    """w of the SSVI slice with ATM total variance theta > 0."""
    z = eta * theta ** (1.0 - gamma) * log_moneyness
    complement = (1.0 - rho) * (1.0 + rho)
    radical = jnp.hypot(z + rho * theta, theta * jnp.sqrt(complement))
    linear = theta + rho * z
    cancelling = linear < 0.0
    size = jnp.where(cancelling, jnp.abs(z), 1.0)
    rationalized = (
        0.5 * complement * size / jnp.where(cancelling, radical / size - linear / size, 1.0)
    )
    return jnp.where(cancelling, rationalized, 0.5 * linear + 0.5 * radical)


def linear_interpolation(times, values, t):
    """values_i + (t - times_i) / (times_{i+1} - times_i) (values_{i+1} - values_i) and the
    slope of the interval [times_i, times_{i+1}) containing t, the last one at times[-1]."""
    index = jnp.clip(jnp.searchsorted(times, t, side="right") - 1, 0, times.size - 2)
    duration = times[index + 1] - times[index]
    increment = values[index + 1] - values[index]
    return values[index] + (t - times[index]) / duration * increment, increment / duration


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["theta_times", "theta_values", "rho", "eta", "gamma"],
    meta_fields=["log_moneyness_bounds"],
)
@dataclass(frozen=True)
class SSVI:
    """Power-law SSVI surface on [0, T_n] x log_moneyness_bounds; see the module docstring.

    theta_times and theta_values are the knots (0, T_1, ..., T_n) and (0, theta_1, ..., theta_n).
    """

    theta_times: jax.Array
    theta_values: jax.Array
    rho: float
    eta: float
    gamma: float
    log_moneyness_bounds: tuple = (-20.0, 20.0)

    def theta(self, t):
        """The ATM total variance theta(t), nan outside [0, T_n]."""
        t = jnp.asarray(t)
        value, _ = linear_interpolation(self.theta_times, self.theta_values, t)
        inside = (t >= 0.0) & (t <= self.theta_times[-1])
        return jnp.where(inside, value, jnp.nan)

    def _inside(self, t, k):
        lower, upper = self.log_moneyness_bounds
        return (t >= 0.0) & (t <= self.theta_times[-1]) & (k >= lower) & (k <= upper)

    def total_variance(self, t, k):
        """w(t, k), zero at t = 0 and nan outside the surface's domain."""
        t, k = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(k, dtype=float))
        positive = self._inside(t, k) & (t > 0.0)
        theta, _ = linear_interpolation(
            self.theta_times, self.theta_values, jnp.where(positive, t, self.theta_times[1])
        )
        w = ssvi_total_variance(theta, jnp.where(positive, k, 0.0), self.rho, self.eta, self.gamma)
        return jnp.where(positive, w, jnp.where(self._inside(t, k), 0.0, jnp.nan))

    def implied_volatility(self, t, k):
        """sqrt(w(t, k) / t) for t > 0."""
        t = jnp.asarray(t, dtype=float)
        return jnp.where(t > 0.0, jnp.sqrt(self.total_variance(t, k) / t), jnp.nan)

    def jet(self, t, k):
        """(w, w_T, w_k, w_kk) at 0 < t < T_n, with the right derivative in t at the knots."""
        t, k = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(k, dtype=float))
        valid = self._inside(t, k) & (t > 0.0) & (t < self.theta_times[-1])
        theta, slope = linear_interpolation(
            self.theta_times, self.theta_values, jnp.where(valid, t, self.theta_times[1])
        )
        k = jnp.where(valid, k, 0.0)
        rho, eta, gamma = self.rho, self.eta, self.gamma
        theta_phi = eta * theta ** (1.0 - gamma)
        z = theta_phi * k
        complement = (1.0 - rho) * (1.0 + rho)
        radical = jnp.hypot(z + rho * theta, theta * jnp.sqrt(complement))
        w = ssvi_total_variance(theta, k, rho, eta, gamma)
        # B = rho + s with s = (z + rho theta) / R, rationalized where rho and s cancel.
        s = (z + rho * theta) / radical
        cancelling = rho * s < 0.0
        rationalized = (
            complement * (z / radical) * ((z + 2.0 * rho * theta) / radical)
        ) / jnp.where(cancelling, s - rho, 1.0)
        b = jnp.where(cancelling, rationalized, rho + s)
        w_t = (w - 0.5 * gamma * z * b) * (slope / theta)
        w_k = 0.5 * theta_phi * b
        w_kk = 0.5 * eta**2 * theta ** (1.0 - 2.0 * gamma) * complement * (theta / radical) ** 3
        return tuple(jnp.where(valid, value, jnp.nan) for value in (w, w_t, w_k, w_kk))

    def initial_local_variance(self, k):
        """lim_{t -> 0} of Dupire's local variance, which exists at k = 0 only:
        (theta_1 / T_1) / g_0 with g_0 = 1 + eta^2 (1 - 2 rho^2) / 4 for gamma = 1/2 and 1
        otherwise."""
        rate = self.theta_values[1] / self.theta_times[1]
        g = jnp.where(self.gamma == 0.5, 1.0 + self.eta**2 * (1.0 - 2.0 * self.rho**2) / 4, 1.0)
        return jnp.where(jnp.asarray(k) == 0.0, rate / g, jnp.nan)

    def conditions(self):
        """The slacks 4 - theta phi (1 + rho), 4 - theta phi (1 - rho), 4 - theta phi^2 (1 + rho)
        and 4 - theta phi^2 (1 - rho) of Gatheral and Jacquier's conditions at theta_n."""
        theta = self.theta_values[-1]
        linear = self.eta * theta ** (1.0 - self.gamma)
        quadratic = self.eta**2 * theta ** (1.0 - 2.0 * self.gamma)
        rho = self.rho
        return jnp.stack(
            (
                4.0 - linear * (1.0 + rho),
                4.0 - linear * (1.0 - rho),
                4.0 - quadratic * (1.0 + rho),
                4.0 - quadratic * (1.0 - rho),
            )
        )

    def extended(self, maturity):
        """The surface with theta continued linearly beyond T_n to the knot at maturity."""
        t, v = self.theta_times, self.theta_values
        if maturity <= float(t[-1]):
            return self
        slope = (v[-1] - v[-2]) / (t[-1] - t[-2])
        return SSVI(
            jnp.append(t, maturity),
            jnp.append(v, v[-1] + slope * (maturity - t[-1])),
            self.rho,
            self.eta,
            self.gamma,
            self.log_moneyness_bounds,
        )
