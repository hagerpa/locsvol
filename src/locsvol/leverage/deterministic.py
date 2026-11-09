"""The deterministic leverage factor of a pure stochastic volatility model."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.functions import compiled


@partial(jax.tree_util.register_dataclass, data_fields=["variance"], meta_fields=[])
@dataclass(frozen=True)
class Deterministic:
    """L(t, x) = sqrt(xi(t)) for a forward variance curve xi: in the LSV model, this leverage
    factor gives the pure stochastic volatility model with variance xi(t) Y_t, for instance the
    backbone as it was calibrated, with xi = 1 the backbone itself."""

    variance: object

    def __post_init__(self):
        object.__setattr__(self, "variance", compiled(self.variance))

    def __call__(self, t, x):
        t, x = jnp.asarray(t), jnp.asarray(x)
        shape = jnp.broadcast_shapes(t.shape, x.shape)
        return jnp.broadcast_to(jnp.sqrt(self.variance(t)), shape)
