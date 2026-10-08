"""The extended SSVI surface (Hendriks and Martini, The extended SSVI volatility surface, J.
Comput. Finance 22, 2019) in the form of Farkas, Ferrari and Ulrych (2022).

The total variance at maturity T and log-moneyness k = log(K/F) is

    w(T, k) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)),

with theta = theta(T) the ATM total variance, linear between knots (0, 0), (T_1, theta_1), ...,
(T_n, theta_n), and with the correlation and the curvature functions of theta,

    phi(theta) = eta theta^-lambda (1 + theta)^(lambda - 1),
    rho(theta) = rho_m + (rho_0 - rho_m) exp(-a theta).

Every slice is an SVI slice with the SSVI shape, so w, w_k and w_kk are those of ssvi.py with
theta phi and rho of the slice: with z = theta phi k, R = sqrt((z + rho theta)^2 + (1 - rho^2)
theta^2) and B = rho + (z + rho theta) / R,

    w = (theta + rho z)/2 + R/2,   w_k = theta phi B / 2,   w_kk = (theta phi)^2 (1 - rho^2) theta^2 / (2 R^3),

and the derivative in T follows from the chain rule through theta, with the right derivative
theta'(T) at the knots:

    w_T = theta'(T)/2 (1 + rho' z + z' B + s (rho' theta + rho) + ((1 - rho^2) theta - rho rho' theta^2) / R),

where s = (z + rho theta) / R, z' = (theta phi)' k with (theta phi)' = theta phi (1 - lambda) /
(theta (1 + theta)) and rho' = -a (rho_0 - rho_m) exp(-a theta). Where theta + rho z or rho and s
cancel, the rationalized forms of ssvi.py are used.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.targets.ssvi import linear_interpolation


def slice_total_variance(theta, z, rho):
    """w of the SSVI-shaped slice with ATM total variance theta > 0 at z = theta phi k."""
    complement = (1.0 - rho) * (1.0 + rho)
    radical = jnp.hypot(z + rho * theta, theta * jnp.sqrt(complement))
    linear = theta + rho * z
    cancelling = linear < 0.0
    size = jnp.where(cancelling, jnp.abs(z), 1.0)
    rationalized = (
        0.5 * complement * size / jnp.where(cancelling, radical / size - linear / size, 1.0)
    )
    return jnp.where(cancelling, rationalized, 0.5 * linear + 0.5 * radical)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["theta_times", "theta_values", "rho_0", "rho_m", "a", "eta", "lam"],
    meta_fields=["log_moneyness_bounds"],
)
@dataclass(frozen=True)
class ESSVI:
    """Extended SSVI surface on [0, T_n] x log_moneyness_bounds; see the module docstring.

    theta_times and theta_values are the knots (0, T_1, ..., T_n) and (0, theta_1, ..., theta_n).
    """

    theta_times: jax.Array
    theta_values: jax.Array
    rho_0: float
    rho_m: float
    a: float
    eta: float
    lam: float
    log_moneyness_bounds: tuple = (-20.0, 20.0)

    @property
    def expiries(self):
        """The knots T_1, ..., T_n."""
        return self.theta_times[1:]

    def theta(self, t):
        """The ATM total variance theta(t), nan outside [0, T_n]."""
        t = jnp.asarray(t)
        value, _ = linear_interpolation(self.theta_times, self.theta_values, t)
        inside = (t >= 0.0) & (t <= self.theta_times[-1])
        return jnp.where(inside, value, jnp.nan)

    def rho(self, theta):
        """rho(theta)."""
        return self.rho_m + (self.rho_0 - self.rho_m) * jnp.exp(-self.a * theta)

    def theta_phi(self, theta):
        """theta phi(theta) = eta theta^(1 - lambda) (1 + theta)^(lambda - 1)."""
        return self.eta * theta ** (1.0 - self.lam) * (1.0 + theta) ** (self.lam - 1.0)

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
        z = self.theta_phi(theta) * jnp.where(positive, k, 0.0)
        w = slice_total_variance(theta, z, self.rho(theta))
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
        rho = self.rho(theta)
        rho_prime = -self.a * (self.rho_0 - self.rho_m) * jnp.exp(-self.a * theta)
        theta_phi = self.theta_phi(theta)
        theta_phi_prime = theta_phi * (1.0 - self.lam) / (theta * (1.0 + theta))
        z, z_prime = theta_phi * k, theta_phi_prime * k
        complement = (1.0 - rho) * (1.0 + rho)
        radical = jnp.hypot(z + rho * theta, theta * jnp.sqrt(complement))
        w = slice_total_variance(theta, z, rho)
        # B = rho + s with s = (z + rho theta) / R, rationalized where rho and s cancel.
        s = (z + rho * theta) / radical
        cancelling = rho * s < 0.0
        rationalized = (
            complement * (z / radical) * ((z + 2.0 * rho * theta) / radical)
        ) / jnp.where(cancelling, s - rho, 1.0)
        b = jnp.where(cancelling, rationalized, rho + s)
        w_theta = 0.5 * (
            1.0
            + rho_prime * z
            + z_prime * b
            + s * (rho_prime * theta + rho)
            + (complement * theta - rho * rho_prime * theta**2) / radical
        )
        w_t = slope * w_theta
        w_k = 0.5 * theta_phi * b
        w_kk = 0.5 * theta_phi**2 * complement * theta**2 / radical**3
        return tuple(jnp.where(valid, value, jnp.nan) for value in (w, w_t, w_k, w_kk))

    def initial_local_variance(self, k):
        """lim_{t -> 0} of Dupire's local variance, which exists at k = 0 only:
        (theta_1 / T_1) / g_0 with g_0 = 1 + eta^2 (1 - 2 rho_0^2) / 4 for lambda = 1/2 and 1
        otherwise, as for SSVI with rho_0 in place of rho."""
        rate = self.theta_values[1] / self.theta_times[1]
        g = jnp.where(self.lam == 0.5, 1.0 + self.eta**2 * (1.0 - 2.0 * self.rho_0**2) / 4, 1.0)
        return jnp.where(jnp.asarray(k) == 0.0, rate / g, jnp.nan)

    def extended(self, maturity):
        """The surface with theta continued linearly beyond T_n to the knot at maturity."""
        t, v = self.theta_times, self.theta_values
        if maturity <= float(t[-1]):
            return self
        slope = (v[-1] - v[-2]) / (t[-1] - t[-2])
        return ESSVI(
            jnp.append(t, maturity),
            jnp.append(v, v[-1] + slope * (maturity - t[-1])),
            self.rho_0,
            self.rho_m,
            self.a,
            self.eta,
            self.lam,
            self.log_moneyness_bounds,
        )
