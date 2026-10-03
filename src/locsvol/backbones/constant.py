"""The constant backbone.

The variance V_t = sigma^2 is constant, so the variance factor Y_t = V_t / E[V_t] = 1 and the local
stochastic volatility model is the local volatility model: the calibrated leverage factor is
the local volatility sigma_D itself. The spot is driven by one Brownian motion.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.tree_util.register_dataclass, data_fields=["level"], meta_fields=[])
@dataclass(frozen=True)
class Constant:
    """Constant volatility level sigma; Y = 1."""

    level: float = 1.0

    factors = 1

    def variance(self, times, increments):
        """Y_t = 1 at all nodes (paths, nodes)."""
        return jnp.ones((increments.shape[0], jnp.size(times)))
