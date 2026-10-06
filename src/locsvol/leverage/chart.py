"""Coordinates of (t, x) for the basis of the leverage factor, x = log(S/F).

From an SSVI surface with ATM total variance theta(t) and parameters eta and gamma (Gatheral and
Jacquier, Quant. Finance 14, 2014), let

    u(t, x) = (2 / pi) arctan(x / s(t)),    s(t) = theta(max(t, t_min))^gamma / eta,

so that s(t) is the log-moneyness scale of the smile at time t and u lies in (-1, 1). The time
coordinate maps s(t) affinely from [s(t_lo), s(t_hi)] onto [-1, 1], clipped there, and is held
at its value at t_clamp below t_clamp. With an envelope [u_lo(t), u_hi(t)], interpolated linearly
in t, the state coordinate is u clipped to the envelope and mapped affinely onto [-1, 1],

    v(t, x) = (2 clip(u, u_lo, u_hi) - u_hi - u_lo) / (u_hi - u_lo),

and v = 0 where u_lo = u_hi; without an envelope it is u.

quantile_bounds gives the quantiles x_eps(t) and x_{1-eps}(t) of a surface, P(X_t <= x_eps(t)) = eps
under the law of X_t that the surface's prices imply (targets.quantiles), and quantile_envelope
their images u(t, x_eps(t)) and u(t, x_{1-eps}(t)).
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.targets.quantiles import digital_quantiles


def linear_theta(times, values, t):
    """Linear interpolation of the ATM total variance, nan outside [times[0], times[-1]]."""
    finite = jnp.isfinite(t)
    safe = jnp.where(finite, t, times[0])
    index = jnp.clip(jnp.searchsorted(times, safe, side="right") - 1, 0, times.shape[0] - 2)
    slope = (values[index + 1] - values[index]) / (times[index + 1] - times[index])
    value = values[index] + slope * (safe - times[index])
    valid = finite & (t >= times[0]) & (t <= times[-1]) & jnp.isfinite(value)
    return jnp.where(valid, value, jnp.nan)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["theta_times", "theta_values", "eta", "gamma", "envelope"],
    meta_fields=["time_bounds", "time_floor", "time_clamp"],
)
@dataclass(frozen=True)
class Chart:
    """The coordinates (tau, u) of the module docstring."""

    theta_times: jax.Array
    theta_values: jax.Array
    eta: float
    gamma: float
    time_bounds: tuple = (0.0, 1.1)
    time_floor: float = 1e-3
    time_clamp: float | None = None
    envelope: tuple | None = None  # (times, lower, upper) of the state coordinate

    def theta(self, t):
        """The ATM total variance theta(max(t, t_min)) of the reference surface."""
        theta = linear_theta(self.theta_times, self.theta_values, jnp.maximum(t, self.time_floor))
        valid = jnp.isfinite(t) & (t >= 0) & (theta > 0) & (t <= self.theta_times[-1])
        return jnp.where(valid, theta, jnp.nan)

    def scale(self, t):
        """s(t) = theta(max(t, t_min))^gamma / eta."""
        return self.theta(t) ** self.gamma / self.eta

    def log_moneyness(self, t, u):
        """The inverse of the state coordinate: x = s(t) tan(pi u / 2)."""
        return self.scale(t) * jnp.tan(0.5 * jnp.pi * u)

    def _affine(self, value, lower, upper):
        return jnp.clip(2 * (value - lower) / (upper - lower) - 1, -1.0, 1.0)

    def time(self, t):
        """The time coordinate tau(t)."""
        return self(t, jnp.asarray(0.0))[..., 0]

    def __call__(self, t, x):
        """(tau, u) at broadcast t and x, shape (..., 2); nan where undefined."""
        t, x = jnp.asarray(t), jnp.asarray(x)
        x = jnp.broadcast_to(x, jnp.broadcast_shapes(t.shape, x.shape))
        valid = jnp.isfinite(t) & (t >= 0) & jnp.isfinite(x)
        scale = self.scale(t)
        state = (2 / jnp.pi) * jnp.arctan(x / scale)
        valid &= jnp.isfinite(scale)
        lower, upper = self.scale(self.time_bounds[0]), self.scale(self.time_bounds[1])
        time = self._affine(scale, lower, upper)
        if self.time_clamp is not None:
            time = jnp.maximum(time, self._affine(self.scale(self.time_clamp), lower, upper))
        valid &= jnp.isfinite(lower) & jnp.isfinite(upper) & (upper > lower)
        values = jnp.stack(jnp.broadcast_arrays(time, state), axis=-1)
        values = jnp.where(valid[..., None] & jnp.isfinite(values), values, jnp.nan)
        if self.envelope is None:
            return values
        times, lower, upper = self.envelope
        lower, upper = jnp.interp(t, times, lower), jnp.interp(t, times, upper)
        width = upper - lower
        clipped = jnp.clip(values[..., -1], lower, upper)
        state = jnp.where(
            width > 0, (2 * clipped - upper - lower) / jnp.where(width > 0, width, 1.0), 0.0
        )
        return values.at[..., -1].set(jnp.where(jnp.isfinite(values[..., -1]), state, jnp.nan))

    def with_envelope(self, times, lower, upper, time_clamp=None):
        """The same chart with an envelope of the state coordinate and a time clamp."""
        return Chart(
            self.theta_times,
            self.theta_values,
            self.eta,
            self.gamma,
            self.time_bounds,
            self.time_floor,
            time_clamp,
            (jnp.asarray(times), jnp.asarray(lower), jnp.asarray(upper)),
        )


def quantile_bounds(surface, times, level, bracket=(-20.0, 20.0)):
    """(times, x_eps, x_{1-eps}): the quantiles of the surface at eps = level and 1 - eps at the
    times; both are 0 at t = 0."""
    times = jnp.asarray(times, dtype=float)
    positive = times > 0
    inner = jnp.where(positive, times, 1.0)
    brackets = jnp.tile(jnp.asarray(bracket, dtype=float), (times.size, 1))
    x = digital_quantiles(surface, inner, jnp.asarray([level, 1.0 - level]), brackets)
    x = jnp.where(positive[:, None], x, 0.0)
    return times, x[:, 0], x[:, 1]


def quantile_envelope(chart, surface, times, level, bracket=(-20.0, 20.0)):
    """Envelope (times, lower, upper) at the images u(t, x_eps(t)) and u(t, x_{1-eps}(t)) of the
    quantiles of the surface at eps = level."""
    times, lower, upper = quantile_bounds(surface, times, level, bracket)
    scale = chart.scale(times)
    return times, (2 / jnp.pi) * jnp.arctan(lower / scale), (2 / jnp.pi) * jnp.arctan(upper / scale)
