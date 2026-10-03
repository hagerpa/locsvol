"""Coordinates of (t, x), x = log(S/F), besides the SSVI chart of chart.py.

Affine:  tau = (t - c_t) / s_t and u = (x - c_x) / s_x, or u alone.
Bounded: u = (2 / pi) arctan((x - c) / s) in (-1, 1) and, given time bounds [t_lo, t_hi], the
         time coordinate tau = 2 (t - t_lo) / (t_hi - t_lo) - 1, clipped to [-1, 1] ('clamp') or
         undefined outside [t_lo, t_hi] ('reject'); without time bounds u alone.
Custom:  a function (t, x) -> (..., d) of the caller on a rectangle [t_lo, t_hi] x [x_lo, x_hi],
         undefined outside it.

The coordinates are undefined (nan) for t < 0 and wherever t or x is not finite. The state
coordinate, the last one, is inverted by x = c_x + s_x u, by x = c + s tan(pi u / 2), and for a
custom chart, whose state coordinate must increase in x, by bisection over [x_lo, x_hi].
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

BISECTION_STEPS = 60  # halvings that resolve any interval of double precision numbers


def _queries(t, x):
    """t, x broadcast for the state, and the queries with finite t >= 0 and finite x."""
    t, x = jnp.asarray(t), jnp.asarray(x)
    x = jnp.broadcast_to(x, jnp.broadcast_shapes(t.shape, x.shape))
    return t, x, jnp.isfinite(t) & (t >= 0) & jnp.isfinite(x)


@partial(
    jax.tree_util.register_dataclass, data_fields=[], meta_fields=["centers", "scales", "with_time"]
)
@dataclass(frozen=True)
class Affine:
    """(tau, u) = ((t - c_t) / s_t, (x - c_x) / s_x), or (u,) without time."""

    centers: tuple[float, float]
    scales: tuple[float, float]
    with_time: bool = True

    def __call__(self, t, x):
        t, x, valid = _queries(t, x)
        state = (x - self.centers[1]) / self.scales[1]
        if self.with_time:
            time = (t - self.centers[0]) / self.scales[0]
            values = jnp.stack(jnp.broadcast_arrays(time, state), axis=-1)
        else:
            values = state[..., None]
        return jnp.where(valid[..., None] & jnp.isfinite(values), values, jnp.nan)

    def log_moneyness(self, t, u):
        """The inverse of the state coordinate, x = c_x + s_x u."""
        return self.centers[1] + self.scales[1] * jnp.asarray(u)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["center", "scale", "time_bounds", "boundary"],
)
@dataclass(frozen=True)
class Bounded:
    """(tau, u) with u = (2 / pi) arctan((x - c) / s); see the module docstring."""

    center: float
    scale: float
    time_bounds: tuple[float, float] | None = None
    boundary: str = "clamp"

    def __post_init__(self):
        if self.boundary not in ("clamp", "reject"):
            raise ValueError("the time boundary must be 'clamp' or 'reject'")

    def __call__(self, t, x):
        t, x, valid = _queries(t, x)
        state = (2 / jnp.pi) * jnp.arctan((x - self.center) / self.scale)
        if self.time_bounds is None:
            values = state[..., None]
        else:
            lower, upper = self.time_bounds
            time = 2 * (t - lower) / (upper - lower) - 1
            if self.boundary == "clamp":
                time = jnp.clip(time, -1.0, 1.0)
            else:
                valid &= (t >= lower) & (t <= upper)
            values = jnp.stack(jnp.broadcast_arrays(time, state), axis=-1)
        return jnp.where(valid[..., None], values, jnp.nan)

    def log_moneyness(self, t, u):
        """The inverse of the state coordinate, x = c + s tan(pi u / 2)."""
        return self.center + self.scale * jnp.tan(0.5 * jnp.pi * jnp.asarray(u))


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["function", "time_bounds", "state_bounds"],
)
@dataclass(frozen=True)
class CustomChart:
    """The coordinates function(t, x) of shape (..., d) on [t_lo, t_hi] x [x_lo, x_hi]."""

    function: object
    time_bounds: tuple[float, float]
    state_bounds: tuple[float, float]

    def __call__(self, t, x):
        t, x, valid = _queries(t, x)
        t = jnp.broadcast_to(t, x.shape)
        (t_lo, t_hi), (x_lo, x_hi) = self.time_bounds, self.state_bounds
        valid &= (t >= t_lo) & (t <= t_hi) & (x >= x_lo) & (x <= x_hi)
        values = self.function(
            jnp.where(valid, t, 0.5 * t_lo + 0.5 * t_hi),
            jnp.where(valid, x, 0.5 * x_lo + 0.5 * x_hi),
        )
        return jnp.where(valid[..., None] & jnp.isfinite(values), values, jnp.nan)

    def log_moneyness(self, t, u):
        """The x in [x_lo, x_hi] whose state coordinate is u, by bisection."""
        t, u = jnp.broadcast_arrays(jnp.asarray(t), jnp.asarray(u))

        def halve(bounds, _):
            low, high = bounds
            middle = 0.5 * (low + high)
            below = self(t, middle)[..., -1] < u
            return (jnp.where(below, middle, low), jnp.where(below, high, middle)), None

        lower, upper = self.state_bounds
        start = (jnp.full(u.shape, lower), jnp.full(u.shape, upper))
        (low, high), _ = jax.lax.scan(halve, start, None, length=BISECTION_STEPS)
        return 0.5 * (low + high)
