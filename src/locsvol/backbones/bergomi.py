"""The two-factor Bergomi model as backbone (Bergomi, Smile dynamics III, 2008).

The variance factor is

    Y_t = xi(t) exp(2 nu / sqrt(q) ((1 - theta) X^1_t + theta X^2_t) - v(t) / 2),

with the Ornstein-Uhlenbeck factors dX^a_t = -k_a X^a_t dt + dZ^a_t, X^a_0 = 0, Brownian motions
with d<W, Z^a>_t = rho_a dt and d<Z^1, Z^2>_t = rho_Z dt, the constant
q = (1 - theta)^2 + theta^2 + 2 rho_Z theta (1 - theta), and v(t) the variance of the exponent,
so that E[Y_t] = xi(t). The factors are simulated exactly: over a step of length h,
X^a_{t+h} = e^{-k_a h} X^a_t + I^a with I^a = int_t^{t+h} e^{-k_a (t+h-s)} dZ^a_s, and the vector
(W_{t+h} - W_t, I^1, I^2) is Gaussian with covariance

    [[h,              rho_1 J(k_1),        rho_2 J(k_2)       ],
     [rho_1 J(k_1),   J(2 k_1),            rho_Z J(k_1 + k_2) ],
     [rho_2 J(k_2),   rho_Z J(k_1 + k_2),  J(2 k_2)           ]],    J(k) = (1 - e^{-k h}) / k.

It is generated from three independent increments by the Cholesky factor of this matrix, whose
first row leaves the spot's increment unchanged.

In the local stochastic volatility model, the variance factor is floored at floor^2 xi(t) and
divided by its exact mean E[max(Y_t, floor^2 xi(t))], a lognormal expectation, so that
E[Y_t] = 1 and the leverage factor carries the level of the volatility.
"""

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr


@partial(jax.tree_util.register_dataclass, data_fields=["boundaries", "values"], meta_fields=[])
@dataclass(frozen=True)
class ForwardVariance:
    """Piecewise constant forward variance: values[i] up to boundaries[i], the last one beyond."""

    boundaries: jax.Array
    values: jax.Array

    def __call__(self, t):
        index = jnp.searchsorted(self.boundaries, t, side="right")
        value = self.values[jnp.minimum(index, self.values.size - 1)]
        return jnp.where(jnp.isfinite(t) & (t >= 0), value, jnp.nan)


def flat_forward_variance(level=1.0):
    """The constant forward variance xi(t) = level."""
    return ForwardVariance(jnp.asarray([jnp.inf]), jnp.asarray([level], dtype=jnp.float64))


def ou_integral(rate, time):
    """int_0^time exp(-rate s) ds."""
    return -jnp.expm1(-rate * time) / rate


def floored_lognormal_mean(log_mean, log_variance, floor):
    """E[max(exp(G), floor)] for a Gaussian G with the given mean and variance."""
    positive = log_variance > 0
    std = jnp.sqrt(jnp.where(positive, log_variance, 1.0))
    log_floor = jnp.log(jnp.where(floor > 0, floor, 1.0))
    below = ndtr((log_floor - log_mean) / std)
    unfloored = jnp.exp(log_mean + 0.5 * log_variance)
    mean = floor * below + unfloored * ndtr((log_mean + log_variance - log_floor) / std)
    mean = jnp.where(positive, mean, jnp.maximum(jnp.exp(log_mean), floor))
    return jnp.where(floor > 0, mean, unfloored)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["nu", "theta", "rho1", "rho2", "k1", "k2", "rho_z", "floor", "forward_variance"],
    meta_fields=[],
)
@dataclass(frozen=True)
class Bergomi:
    """Two-factor Bergomi backbone; see the module docstring."""

    nu: float
    theta: float
    rho1: float
    rho2: float
    k1: float = 4.0
    k2: float = 0.12
    rho_z: float = 0.3
    floor: float = 1e-5
    forward_variance: ForwardVariance = field(default_factory=flat_forward_variance)

    factors = 3

    @property
    def q(self):
        return (
            (1 - self.theta) ** 2 + self.theta**2 + 2 * self.rho_z * self.theta * (1 - self.theta)
        )

    def exponent_variance(self, t):
        """v(t), the variance of 2 nu / sqrt(q) ((1 - theta) X^1_t + theta X^2_t)."""
        integrated = (1 - self.theta) ** 2 * ou_integral(2 * self.k1, t)
        integrated += self.theta**2 * ou_integral(2 * self.k2, t)
        integrated += (
            2 * self.rho_z * self.theta * (1 - self.theta) * ou_integral(self.k1 + self.k2, t)
        )
        return 4 * self.nu**2 / self.q * integrated

    def transition(self, h):
        """Decay e^{-k_a h} and the map from independent increments to (I^1, I^2)."""
        a, b = self.rho1 * ou_integral(self.k1, h), self.rho2 * ou_integral(self.k2, h)
        c = self.rho_z * ou_integral(self.k1 + self.k2, h)
        covariance = jnp.array(
            [[h, a, b], [a, ou_integral(2 * self.k1, h), c], [b, c, ou_integral(2 * self.k2, h)]]
        )
        cholesky = jnp.where(h > 0, jnp.linalg.cholesky(covariance), jnp.nan)
        return jnp.exp(-jnp.array([self.k1, self.k2]) * h), cholesky[1:] / jnp.sqrt(h)

    def raw_variance(self, factors, t):
        """xi(t) max(exp(...), floor^2) for the factor values (..., 2) at time t."""
        exponent = (
            2
            * self.nu
            / jnp.sqrt(self.q)
            * ((1 - self.theta) * factors[..., 0] + self.theta * factors[..., 1])
        )
        multiplier = jnp.exp(exponent - 0.5 * self.exponent_variance(t))
        return self.forward_variance(t) * jnp.maximum(multiplier, self.floor**2)

    def mean(self, t):
        """E[raw_variance] at the times t."""
        variance = self.exponent_variance(t)
        forward = self.forward_variance(t)
        return floored_lognormal_mean(
            -0.5 * variance + jnp.log(forward), variance, self.floor**2 * forward
        )

    @jax.jit
    def variance(self, times, increments):
        """Y_t / E[Y_t] at all nodes (paths, nodes) from increments (paths, steps, 3)."""
        decay, innovation = jax.vmap(self.transition)(jnp.diff(times))

        def step(state, j):
            value = self.raw_variance(state, times[j])
            dw = increments[:, j, :]
            following = decay[j] * state + jnp.einsum("pd,rd->pr", dw, innovation[j])
            return following, value

        start = jnp.zeros((increments.shape[0], 2))
        final, values = jax.lax.scan(step, start, jnp.arange(times.size - 1))
        raw = jnp.concatenate((values.T, self.raw_variance(final, times[-1])[:, None]), axis=1)
        return raw / self.mean(times)[None, :]
