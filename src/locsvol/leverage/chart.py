"""Coordinates of (t, x) for the basis of the leverage factor, x = log(S/F).

From an SSVI surface with ATM total variance theta(t) and parameters eta and gamma (Gatheral and
Jacquier, Quant. Finance 14, 2014), the state coordinate is

    u(t, x) = (2 / pi) arctan(x / s(t)),    s(t) = theta(max(t, t_min))^gamma / eta,

so that s(t) is the log-moneyness scale of the smile at time t and u lies in (-1, 1). The time
coordinate maps s(t) affinely from [s(t_lo), s(t_hi)] onto [-1, 1], clipped there, and is held
at its value at t_clamp below t_clamp. An envelope [lower(t), upper(t)], interpolated linearly in
t, clips u to the states that the simulated paths reach.

The envelope is read off simulated paths: at each node, the quantile of |u| at a probability
close to one, widened to its running maximum in t and capped at the image of a fixed number of
ATM standard deviations, x = +- c sqrt(theta(t)).
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


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
        clipped = jnp.clip(
            values[..., -1], jnp.interp(t, times, lower), jnp.interp(t, times, upper)
        )
        return values.at[..., -1].set(jnp.where(jnp.isfinite(values[..., -1]), clipped, jnp.nan))

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


def envelope(chart, times, log_spot, quantile=0.999, running_maximum=True, cap_sigmas=9.2):
    """Envelope (times, lower, upper) of the state coordinate from paths X (paths, nodes)."""
    state = chart(jnp.broadcast_to(times[None, :], log_spot.shape), log_spot)[..., -1]
    width = jnp.quantile(jnp.abs(state), quantile, axis=0)
    lower, upper = -width, width
    if running_maximum:
        lower, upper = jnp.minimum.accumulate(lower), jnp.maximum.accumulate(upper)
    if cap_sigmas is not None:
        x = cap_sigmas * jnp.sqrt(chart.theta(times))
        floor, ceiling = chart(jnp.stack((times, times)), jnp.stack((-x, x)))[..., -1]
        lower, upper = jnp.maximum(lower, floor), jnp.minimum(upper, ceiling)
    return times, lower, upper
