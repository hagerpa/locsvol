"""The SABR model with beta = 1 as backbone (Hagan, Kumar, Lesniewski and Woodward, Managing smile
risk, Wilmott Magazine, 2002).

The volatility alpha_t = alpha_0 exp(l_t) is lognormal,

    dl_t = nu dZ_t - nu^2 dt / 2,   l_0 = 0,   d<W, Z>_t = rho dt,

and is simulated exactly: over a step of length h,

    l_{t+h} = l_t + nu (rho dW + sqrt(1 - rho^2) dW') - nu^2 h / 2,

with the spot's increment dW and an independent increment dW'. In the local stochastic volatility
model the variance factor is V_t = max(alpha_t, floor)^2 divided by its exact mean, a floored
lognormal expectation with log alpha_t^2 ~ N(2 log alpha_0 - nu^2 t, 4 nu^2 t), so that
E[Y_t] = 1.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.backbones.bergomi import floored_lognormal_mean


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["alpha", "rho", "nu", "floor"],
    meta_fields=[],
)
@dataclass(frozen=True)
class SABR:
    """SABR backbone with beta = 1; see the module docstring."""

    alpha: float
    rho: float
    nu: float
    floor: float = 0.0

    factors = 2

    def step(self, ell, h, dw):
        """l_{t+h} from l_t = ell and the independent increments dw (..., 2)."""
        noise = self.rho * dw[..., 0] + jnp.sqrt(1.0 - jnp.square(self.rho)) * dw[..., 1]
        return ell + self.nu * noise - 0.5 * jnp.square(self.nu) * h

    def raw_variance(self, ell):
        """max(alpha_0 exp(ell), floor)^2."""
        return jnp.maximum(self.alpha * jnp.exp(ell), self.floor) ** 2

    def mean(self, t):
        """E[max(alpha_t, floor)^2] at the times t."""
        return floored_lognormal_mean(
            2 * jnp.log(self.alpha) - self.nu**2 * t, 4 * self.nu**2 * t, self.floor**2
        )

    @jax.jit
    def variance(self, times, increments):
        """Y_t at all nodes (paths, nodes) from increments (paths, steps, 2)."""

        def step(ell, j):
            following = self.step(ell, times[j + 1] - times[j], increments[:, j, :])
            return following, self.raw_variance(ell)

        final, values = jax.lax.scan(
            step, jnp.zeros(increments.shape[0]), jnp.arange(times.size - 1)
        )
        values = jnp.concatenate((values.T, self.raw_variance(final)[:, None]), axis=1)
        return values / self.mean(times)[None, :]
