"""Backbones of the caller.

A backbone is any object with the number factors of Brownian motions it uses, the spot's first,
and a method variance(times, increments) that returns the variance factor Y at all nodes
(paths, nodes) from the independent increments (paths, steps, factors). CustomBackbone builds
one from a Markov scheme of the caller,

    s_0 = start(parameters, paths),
    s_{j+1} = transition(parameters, s_j, t_j, t_{j+1} - t_j, dW_j),
    V_j = state_variance(parameters, s_j, t_j),   Y_{t_j} = V_j / mean(parameters, t_j),

with the mean E[V_t] of the scheme, so that E[Y_t] = 1; without a mean, Y = V. The parameters are
a pytree of arrays, so that compiled code can differentiate the variance factor with respect to
them; the functions must be JAX functions.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["parameters"],
    meta_fields=["factors", "start", "transition", "state_variance", "mean"],
)
@dataclass(frozen=True)
class CustomBackbone:
    """The backbone of a Markov scheme of the caller; see the module docstring."""

    factors: int
    start: object
    transition: object
    state_variance: object
    mean: object = None
    parameters: object = None

    @jax.jit
    def variance(self, times, increments):
        """Y_t at all nodes (paths, nodes) from increments (paths, steps, factors)."""
        p = self.parameters

        def step(state, j):
            dt = times[j + 1] - times[j]
            following = self.transition(p, state, times[j], dt, increments[:, j, :])
            return following, self.state_variance(p, state, times[j])

        state = self.start(p, increments.shape[0])
        final, values = jax.lax.scan(step, state, jnp.arange(times.size - 1))
        final = self.state_variance(p, final, times[-1])
        variance = jnp.concatenate((values.T, final[:, None]), axis=1)
        return variance if self.mean is None else variance / self.mean(p, times)[None, :]
