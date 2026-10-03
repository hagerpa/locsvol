"""The Heston model as backbone (Heston, Rev. Financ. Stud. 6, 1993).

The variance factor follows the CIR process

    dV_t = kappa (theta - V_t) dt + sigma sqrt(V_t) dZ_t,   V_0 = v0,   d<W, Z>_t = rho dt,

and is simulated by the quadratic-exponential scheme of Andersen (J. Comput. Finance 11, 2008),
which matches the exact conditional mean m and variance s^2 of V_{t+h} given V_t: with
psi = s^2 / m^2, it samples a squared shifted normal for psi <= 1.5 and an exponential law with
an atom at zero otherwise, both driven by the normal Z_h of the correlated increment
rho dW + sqrt(1 - rho^2) dW', with uniform Phi(Z_h) in the exponential branch. The variance is
floored at floor after each step.

In the local stochastic volatility model, Y_t = V_t / E[max(V_t, floor)], with the exact floored
mean of the noncentral chi-square law of V_t, so that E[Y_t] = 1.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln, ndtr

SWITCHING_LEVEL = 1.5
POISSON_TERMS = 64


def noncentral_chi_square_cdf(x, degrees, noncentrality):
    """P(X <= x) for a noncentral chi-square law, as a Poisson mixture of central ones."""
    x, degrees, noncentrality = (
        jnp.asarray(v, dtype=jnp.float64) for v in (x, degrees, noncentrality)
    )
    half = 0.5 * noncentrality
    j = jnp.arange(POISSON_TERMS, dtype=jnp.float64)
    shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(degrees), jnp.shape(half))
    log_half = jnp.log(jnp.maximum(half, 1e-300))[..., None]
    weight = jnp.exp(-half[..., None] + j * log_half - gammaln(j + 1.0))
    component = gammainc(0.5 * degrees[..., None] + j, 0.5 * jnp.broadcast_to(x, shape)[..., None])
    return jnp.sum(weight * component, axis=-1)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["kappa", "theta", "sigma", "rho", "v0", "floor"],
    meta_fields=[],
)
@dataclass(frozen=True)
class Heston:
    """Heston backbone; see the module docstring."""

    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float
    floor: float = 1e-8

    factors = 2

    def step(self, v, h, dw):
        """V_{t+h} from V_t by the quadratic-exponential scheme, increments dw (..., 2)."""
        v = jnp.maximum(v, self.floor)
        z = (self.rho * dw[..., 0] + jnp.sqrt(1.0 - jnp.square(self.rho)) * dw[..., 1]) / jnp.sqrt(
            h
        )
        decay = jnp.exp(-self.kappa * h)
        mean = self.theta + (v - self.theta) * decay
        variance = v * self.sigma**2 * decay * (1.0 - decay) / self.kappa
        variance += self.theta * self.sigma**2 * (1.0 - decay) ** 2 / (2.0 * self.kappa)
        psi = variance / jnp.square(mean)
        bounded = jnp.minimum(psi, SWITCHING_LEVEL)
        inverse = 2.0 / bounded
        b_squared = inverse - 1.0 + jnp.sqrt(inverse) * jnp.sqrt(jnp.maximum(inverse - 1.0, 0.0))
        quadratic = (mean / (1.0 + b_squared)) * jnp.square(jnp.sqrt(b_squared) + z)
        atom = (psi - 1.0) / (psi + 1.0)
        rate = (1.0 - atom) / mean
        uniform = ndtr(z)
        exponential = jnp.where(
            uniform <= atom, 0.0, jnp.log((1.0 - atom) / jnp.maximum(1.0 - uniform, 1e-300)) / rate
        )
        value = jnp.where(psi <= SWITCHING_LEVEL, quadratic, exponential)
        return jnp.maximum(value, self.floor)

    def mean(self, t):
        """E[max(V_t, floor)] from the noncentral chi-square law of V_t given v0."""
        decay = jnp.exp(-self.kappa * t)
        unfloored = self.theta + (self.v0 - self.theta) * decay
        positive = t > 0
        scale = (
            2.0
            * self.kappa
            / (self.sigma**2 * jnp.where(positive, -jnp.expm1(-self.kappa * t), 1.0))
        )
        degrees = 4.0 * self.kappa * self.theta / self.sigma**2
        noncentrality = 2.0 * scale * self.v0 * decay
        argument = 2.0 * scale * self.floor
        below = noncentral_chi_square_cdf(argument, degrees, noncentrality)
        partial_mean = noncentral_chi_square_cdf(argument, degrees + 2.0, noncentrality)
        truncated = (degrees + noncentrality) * partial_mean / (2.0 * scale)
        mean = unfloored + self.floor * below - truncated
        return jnp.where(positive, mean, jnp.maximum(self.v0, self.floor))

    @jax.jit
    def variance(self, times, increments):
        """Y_t = V_t / E[max(V_t, floor)] at all nodes (paths, nodes).

        increments has shape (paths, steps, 2).
        """

        def step(v, j):
            following = self.step(v, times[j + 1] - times[j], increments[:, j, :])
            return following, v

        start = jnp.full(increments.shape[0], self.v0)
        final, values = jax.lax.scan(step, start, jnp.arange(times.size - 1))
        raw = jnp.concatenate((values.T, final[:, None]), axis=1)
        return raw / self.mean(times)[None, :]
