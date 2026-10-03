"""The refined SVI surface: a raw SVI slice at every expiry, interpolated in price.

The slices w_i(k) = w(T_i, k) at the expiries T_1 < ... < T_n have increasing ATM total variances
theta_i = w_i(0), and theta(t) is linear between the knots (0, 0), (T_1, theta_1), ...,
(T_n, theta_n). Between T_i and T_{i+1}, the normalized prices of the out-of-the-money options of
the two slices are interpolated linearly in the square root of the ATM total variance,

    BS(k, w(t, k)) = alpha BS(k, w_i(k)) + (1 - alpha) BS(k, w_{i+1}(k)),
    alpha = (sqrt(theta_{i+1}) - sqrt(theta(t))) / (sqrt(theta_{i+1}) - sqrt(theta_i)),

and w(t, k) is the total variance of the interpolated price (inversion.otm_total_variance). Far in
the wings, where |d1| = |-k / sqrt(w_{i+1}) + sqrt(w_{i+1}) / 2| exceeds a resolution limit,
prices no longer determine w, and the surface takes the slice w_{i+1} there (and w_i at T_i).
Beyond the last expiry the total variance grows with a constant forward variance xi,
w(t, k) = w_n(k) + xi (t - T_n). Below the first expiry, the short extension is

    w(t, k) = (t / T_1) (1 + c (1 - t / T_1)) w_1(k sqrt((T_1 + t_0) / (t + t_0)))

with shift t_0 >= 0 and level c. t_0 = 0 gives the self-similar extension and t_0 = inf the
fixed smile shape w_1; c = 0 keeps theta linear, c > 0 moves variance towards t = 0. With a
butterfly-free first slice, the extension is free of butterfly arbitrage for
-1 <= c <= t_0 / (T_1 + t_0). short_extension chooses t_0 and c such that the forward variance
d/dT V(T) of the log-contract is continuous at T_1, with c = 0 whenever possible. For t_0 > 0,
w(t, k) = t g(k) + O(t^2) as t -> 0 with g(k) = (1 + c) w_1(k / lambda_0) / T_1 and
lambda_0 = sqrt(t_0 / (T_1 + t_0)), so that Dupire's local variance tends to
g / (1 - k g' / (2 g))^2.

The derivatives of w in t and k, and the second derivative in k, are computed by automatic
differentiation.
"""

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.pricing.black import black_price
from locsvol.targets.inversion import otm_total_variance
from locsvol.targets.log_contract import log_contract_variance
from locsvol.targets.svi import svi, svi_derivatives


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["expiries", "slices", "shift", "level"],
    meta_fields=[
        "long_forward_variance",
        "maximum_maturity",
        "log_moneyness_bounds",
        "resolution",
    ],
)
@dataclass(frozen=True)
class RefinedSVI:
    """Raw SVI slices (n, 5) at the expiries (n,); see the module docstring.

    shift and level are t_0 and c of the short extension, long_forward_variance is xi beyond the
    last expiry, and resolution is the limit of |d1| below which prices are interpolated. The
    surface is defined on [0, maximum_maturity] x log_moneyness_bounds.
    """

    expiries: jax.Array
    slices: jax.Array
    shift: float = 0.0
    level: float = 0.0
    long_forward_variance: float = 0.04
    maximum_maturity: float = 1.2
    log_moneyness_bounds: tuple = (-20.0, 20.0)
    resolution: float = 26.0

    def atm(self):
        """The ATM total variances theta_i = w_i(0) of the slices."""
        return jax.vmap(svi, in_axes=(0, None))(self.slices, 0.0)

    def _inside(self, t, k):
        lower, upper = self.log_moneyness_bounds
        return (
            jnp.isfinite(t)
            & jnp.isfinite(k)
            & (t >= 0.0)
            & (t <= self.maximum_maturity)
            & (k >= lower)
            & (k <= upper)
        )

    def _scale(self, t):
        """sqrt((t + t_0) / (T_1 + t_0)), one for t_0 = inf."""
        infinite = jnp.isinf(self.shift)
        shift = jnp.where(infinite, 0.0, self.shift)
        return jnp.where(infinite, 1.0, jnp.sqrt((t + shift) / (self.expiries[0] + shift)))

    def _resolved(self, w, k):
        root = jnp.sqrt(w)
        return jnp.abs(-k / root + 0.5 * root) < self.resolution

    def _value(self, t, k):
        """w(t, k) at a single t > 0 and k."""
        expiries, slices = self.expiries, self.slices
        first, last = expiries[0], expiries[-1]
        short, long = t < first, t >= last

        s = jnp.where(short, t, first)
        x = s / first
        short_value = x * (1.0 + self.level * (1.0 - x)) * svi(slices[0], k / self._scale(s))
        long_value = svi(slices[-1], k) + self.long_forward_variance * jnp.where(
            long, t - last, 0.0
        )

        u = jnp.where(~short & ~long, t, first)
        high = jnp.clip(jnp.searchsorted(expiries, u, side="right"), 1, expiries.size - 1)
        low = high - 1
        atm = self.atm()
        times = jnp.concatenate((jnp.zeros(1), expiries))
        values = jnp.concatenate((jnp.zeros(1), atm))
        index = jnp.clip(jnp.searchsorted(times, u, side="right") - 1, 0, times.size - 2)
        slope = (values[index + 1] - values[index]) / (times[index + 1] - times[index])
        theta = values[index] + slope * (u - times[index])
        alpha = (jnp.sqrt(atm[high]) - jnp.sqrt(theta)) / (jnp.sqrt(atm[high]) - jnp.sqrt(atm[low]))
        low_value, high_value = svi(slices[low], k), svi(slices[high], k)
        resolved = self._resolved(high_value, k)
        # Where the surface does not interpolate, the unused inversion runs at k = 0, where it is
        # regular, so that it does not spoil the derivatives.
        kk = jnp.where(resolved & ~short & ~long, k, 0.0)
        low_price = black_price(low_value, kk, kk >= 0.0)
        high_price = black_price(high_value, kk, kk >= 0.0)
        price = alpha * low_price + (1.0 - alpha) * high_price
        interpolated = otm_total_variance(price, kk, low_value, high_value, low_price, high_price)
        wing = jnp.where(u == expiries[low], low_value, high_value)
        interior = jnp.where(resolved, interpolated, wing)
        return jnp.where(short, short_value, jnp.where(long, long_value, interior))

    def _time_derivative(self, t, k):
        """d_t w at a single t > 0, nan at an expiry where the surface jumps in the wing."""
        expiries = self.expiries
        index = jnp.clip(jnp.searchsorted(expiries, t, side="left"), 0, expiries.size - 1)
        following = jnp.minimum(index + 1, expiries.size - 1)
        jump = (
            (t == expiries[index])
            & (index < expiries.size - 1)
            & ~self._resolved(svi(self.slices[following], k), k)
        )
        return jnp.where(jump, jnp.nan, jax.grad(RefinedSVI._value, argnums=1)(self, t, k))

    def total_variance(self, t, k):
        """w(t, k), zero at t = 0 and nan outside the surface's domain."""
        return _total_variance(self, t, k)

    def implied_volatility(self, t, k):
        """sqrt(w(t, k) / t) for t > 0."""
        t = jnp.asarray(t, dtype=float)
        return jnp.where(t > 0.0, jnp.sqrt(self.total_variance(t, k) / t), jnp.nan)

    def jet(self, t, k):
        """(w, w_T, w_k, w_kk) at 0 < t < maximum_maturity, with the right derivative in t at the
        expiries."""
        return _jet(self, t, k)

    def initial_local_variance(self, k):
        """lim_{t -> 0} of Dupire's local variance: g / (1 - k g' / (2 g))^2 for t_0 > 0, and
        (theta_1 / T_1) / (1 - w_1'(0)^2 / (4 theta_1) + w_1''(0) / 2) at k = 0 for t_0 = 0."""
        k = jnp.asarray(k, dtype=float)
        first, first_slice = self.expiries[0], self.slices[0]
        scale = self._scale(0.0)
        y = jnp.where(self.shift > 0.0, k / scale, 0.0)
        value, (slope, _) = svi(first_slice, y), svi_derivatives(first_slice, y)
        factor = jnp.square(1.0 - 0.5 * k * slope / (scale * value))
        row = (1.0 + self.level) * (value / (first * factor))
        row = jnp.where(jnp.isfinite(factor) & (factor > 0.0), row, jnp.nan)
        w, (w_k, w_kk) = svi(first_slice, 0.0), svi_derivatives(first_slice, 0.0)
        origin = (w / first) / (1.0 - w_k**2 / (4.0 * w) + 0.5 * w_kk)
        limit = jnp.where(self.shift > 0.0, row, jnp.where(k == 0.0, origin, jnp.nan))
        return jnp.where(self._inside(jnp.zeros_like(k), k), limit, jnp.nan)


@jax.jit
def _total_variance(surface, t, k):
    t, k = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(k, dtype=float))
    inside = surface._inside(t, k)
    safe_t = jnp.where(inside & (t > 0.0), t, surface.expiries[0])
    safe_k = jnp.where(inside, k, 0.0)
    value = jax.vmap(RefinedSVI._value, in_axes=(None, 0, 0))(
        surface, safe_t.ravel(), safe_k.ravel()
    )
    return jnp.where(inside, jnp.where(t == 0.0, 0.0, value.reshape(t.shape)), jnp.nan)


@jax.jit
def _jet(surface, t, k):
    t, k = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(k, dtype=float))
    valid = surface._inside(t, k) & (t > 0.0) & (t < surface.maximum_maturity)
    safe_t = jnp.where(valid, t, surface.expiries[0]).ravel()
    safe_k = jnp.where(valid, k, 0.0).ravel()
    slope = jax.grad(RefinedSVI._value, argnums=2)
    curvature = jax.grad(slope, argnums=2)
    jet = []
    for function in (RefinedSVI._value, RefinedSVI._time_derivative, slope, curvature):
        value = jax.vmap(function, in_axes=(None, 0, 0))(surface, safe_t, safe_k)
        jet.append(jnp.where(valid, value.reshape(t.shape), jnp.nan))
    return tuple(jet)


def short_extension(
    surface, order=256, bracket=(0.0, 1.0), step=1e-6, tolerance=1e-12, with_level=True
):
    """The surface with t_0 and c such that the log-contract forward variance is continuous at T_1.

    The forward variance below T_1 is the difference quotient (V(T_1) - V(T_1 - step)) / step,
    with V from log_contract_variance on the surface's log-moneyness bounds, and the one above
    T_1 is (V(T_1 + step) - V(T_1)) / step. It increases with t_0 and decreases with c. If a t_0
    in the bracket matches the two at c = 0, it is found by bisection. Otherwise, for with_level,
    the pair with the smallest |c|: if the forward variance below T_1 is too large even at the
    bracket's lower end, (t_0, c) on the butterfly bound c = t_0 / (T_1 + t_0); if it is too small
    at the upper end, t_0 at the upper end and c in [-1, 0). Bisections stop at the tolerance.
    """
    first = float(surface.expiries[0])
    if not 0.0 < step < first:
        raise ValueError("the step must lie inside the first expiry")

    def variance(s, t):
        return float(log_contract_variance(s, jnp.asarray(t), order))

    right = (variance(surface, first + step) - variance(surface, first)) / step

    def jump(shift, c=0.0):
        s = replace(surface, shift=shift, level=c)
        return (variance(s, first) - variance(s, first - step)) / step - right

    low, high = bracket
    low_jump, high_jump = jump(low), jump(high)
    if low_jump * high_jump <= 0.0:
        return replace(surface, shift=_bisect(jump, low, high, tolerance), level=0.0)
    if not with_level:
        raise ValueError(f"no shift in {bracket} makes the forward variance continuous at T_1")
    if low_jump > 0.0:
        shift = _bisect(lambda t0: jump(t0, t0 / (first + t0)), low, high, tolerance)
        return replace(surface, shift=shift, level=shift / (first + shift))
    return replace(
        surface, shift=high, level=_bisect(lambda c: jump(high, c), -1.0, 0.0, tolerance)
    )


def _bisect(function, low, high, tolerance):
    """A root of function in [low, high] by bisection, given values of opposite sign at the ends."""
    low_value, high_value = function(low), function(high)
    if low_value * high_value > 0.0:
        raise ValueError(f"no sign change in [{low}, {high}]: {low_value:.6e}, {high_value:.6e}")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if function(middle) * low_value > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
