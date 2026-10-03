"""Targets with a closed-form local volatility, for tests.

Constant volatility sigma: sigma_Dup(t, x) = sigma and w(t, k) = sigma^2 t.

A term structure of forward variance xi(t), constant on the intervals [t_j, t_{j+1}) of a grid
0 = t_0 < ... < t_m: sigma_Dup(t, x) = sqrt(xi(t)) and w(t, k) = int_0^t xi(s) ds, the same for
every k.

A synthetic local volatility in x = log(S/F): with weights p_i summing to one, which may be
negative, volatilities sigma_i, K_i = exp(-x^2 / (2 t sigma_i^2) - t sigma_i^2 / 8),
A(x) = gamma_0 max(x - beta_0, 0) - gamma_1 min(x + beta_1, 0) and
H(t, x) = min(A(x)^kappa, lambda_0) (1_{t <= 0.1} / (1 + 0.1 t))^lambda_1,

    sigma(t, x)^2 = min(|c(t) (sum_i p_i sigma_i K_i + H) / (sum_i p_i K_i / sigma_i + 0.01)|, 2)/4,

where c(t) = 1 for t <= 0.1 and 0.4 afterwards. At t = 0 it is defined at x = 0 only, where
K_i = 1 and H = 0.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.tree_util.register_dataclass, data_fields=["volatility"], meta_fields=[])
@dataclass(frozen=True)
class ConstantVolatility:
    """sigma_Dup = sigma, w(t, k) = sigma^2 t."""

    volatility: float

    log_moneyness_bounds = (-jnp.inf, jnp.inf)

    def __call__(self, t, x):
        t, x = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(x, dtype=float))
        valid = jnp.isfinite(t) & (t >= 0.0) & jnp.isfinite(x)
        return jnp.where(valid, self.volatility, jnp.nan)

    def total_variance(self, t, k):
        """sigma^2 t."""
        t, k = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(k, dtype=float))
        valid = jnp.isfinite(t) & (t >= 0.0) & jnp.isfinite(k)
        return jnp.where(valid, t * jnp.square(self.volatility), jnp.nan)


@partial(
    jax.tree_util.register_dataclass, data_fields=["times", "forward_variances"], meta_fields=[]
)
@dataclass(frozen=True)
class TermStructure:
    """sigma_Dup(t, x) = sqrt(xi(t)) for xi = forward_variances[j] on [times[j], times[j+1])."""

    times: jax.Array
    forward_variances: jax.Array

    log_moneyness_bounds = (-jnp.inf, jnp.inf)

    def __call__(self, t, x):
        """sqrt(xi(t)) on [0, t_m], the last value at t_m."""
        t, x = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(x, dtype=float))
        index = jnp.searchsorted(self.times, t, side="right") - 1
        index = jnp.clip(index, 0, self.forward_variances.size - 1)
        valid = jnp.isfinite(t) & (t >= 0.0) & (t <= self.times[-1]) & jnp.isfinite(x)
        return jnp.where(valid, jnp.sqrt(self.forward_variances[index]), jnp.nan)

    def total_variance(self, t, k):
        """int_0^t xi(s) ds on [0, t_m]."""
        t, k = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(k, dtype=float))
        totals = jnp.concatenate(
            (jnp.zeros(1), jnp.cumsum(jnp.diff(self.times) * self.forward_variances))
        )
        index = jnp.clip(jnp.searchsorted(self.times, t, side="right") - 1, 0, self.times.size - 2)
        value = totals[index] + self.forward_variances[index] * (t - self.times[index])
        valid = jnp.isfinite(t) & (t >= 0.0) & (t <= self.times[-1]) & jnp.isfinite(k)
        return jnp.where(valid, value, jnp.nan)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["sigmas", "weights", "beta", "gamma", "lam", "kappa"],
    meta_fields=[],
)
@dataclass(frozen=True)
class Synthetic:
    """The synthetic local volatility of the module docstring: sigmas and weights (3,), beta,
    gamma and lam (2,), kappa a scalar."""

    sigmas: jax.Array
    weights: jax.Array
    beta: jax.Array
    gamma: jax.Array
    lam: jax.Array
    kappa: float

    log_moneyness_bounds = (-jnp.inf, jnp.inf)

    def __call__(self, t, x):
        t, x = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(x, dtype=float))
        origin = (t == 0.0) & (x == 0.0)
        valid = jnp.isfinite(t) & jnp.isfinite(x) & ((t > 0.0) | origin)
        t_, x_ = jnp.where(valid & (t > 0.0), t, 1.0)[..., None], jnp.where(valid, x, 0.0)
        kernels = jnp.exp(
            -0.5 * x_[..., None] ** 2 / (t_ * self.sigmas**2) - 0.125 * t_ * self.sigmas**2
        )
        kernels = jnp.where(origin[..., None], 1.0, kernels)
        tail = self.gamma[0] * jnp.maximum(x_ - self.beta[0], 0.0) - self.gamma[1] * jnp.minimum(
            x_ + self.beta[1], 0.0
        )
        positive = tail > 0.0
        tail = jnp.where(positive, jnp.where(positive, tail, 1.0) ** self.kappa, 0.0)
        tail = jnp.minimum(tail, self.lam[0])
        tail *= jnp.where(t_[..., 0] <= 0.1, 1.0 / (1.0 + 0.1 * t_[..., 0]), 0.0) ** self.lam[1]
        numerator = jnp.sum(self.weights * self.sigmas * kernels, axis=-1) + tail
        numerator *= jnp.where((t > 0.1) & ~origin, 0.4, 1.0)
        denominator = jnp.sum(self.weights * kernels / self.sigmas, axis=-1) + 0.01
        ratio = numerator / denominator
        value = jnp.sqrt(0.25 * jnp.minimum(jnp.abs(ratio), 2.0))
        valid &= ~origin | (jnp.isfinite(ratio) & (value > 0.0))
        return jnp.where(valid & jnp.isfinite(value), value, jnp.nan)
