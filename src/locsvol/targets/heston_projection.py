"""The local volatility of a Heston model, by Markovian projection.

In the Heston model dX_t = -V_t/2 dt + sqrt(V_t) dW_t, X = log(S/F), the local volatility

    sigma(t, x)^2 = E[V_t | X_t = x]

gives the local volatility model with the same marginal laws of X (Gyongy, Probab. Theory Relat.
Fields 71, 1986), so the Heston prices are its prices. The conditional mean is estimated by
regression of V_t on coordinates of (t, X_t) over simulated Heston paths: with a regression of
log link such as GammaRegression, sigma(t, x)^2 = exp(b(z(t, x))^T alpha) for coordinates z and
the basis b of the regression.

The paths are those of the backbone: its variance factor Y_t = V_t / E[max(V_t, floor)] times
its mean is V_t, and the log spot follows from the Euler scheme with the leverage factor
sqrt(E[max(V_t, floor)]), so that L^2 Y = V.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.functions import compiled
from locsvol.simulation.paths import simulate


@jax.jit
def _mean(heston, t):
    return heston.mean(t)


@partial(jax.tree_util.register_dataclass, data_fields=["heston"], meta_fields=[])
@dataclass(frozen=True)
class _MeanLeverage:
    """sqrt(E[max(V_t, floor)]), so that the Euler scheme simulates the Heston log spot."""

    heston: object

    def __call__(self, t, x):
        return jnp.sqrt(self.heston.mean(t)) + jnp.zeros_like(x)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["coordinates", "coefficients"],
    meta_fields=["basis"],
)
@dataclass(frozen=True)
class HestonProjection:
    """sigma(t, x) = sqrt(exp(b(z(t, x))^T alpha)) for coordinates z and a basis b."""

    coordinates: object
    basis: object
    coefficients: jax.Array

    def __call__(self, t, x):
        z = self.coordinates(t, x)
        return jnp.sqrt(jnp.exp(self.basis(z) @ self.coefficients))


def heston_projection(heston, times, stream, paths, coordinates, regression, weights=None):
    """The projection of the Heston backbone on its log spot; see the module docstring.

    The paths are simulated on the grid times from the stream as in simulation.simulate, and
    every path and node before the last enters the regression of V on coordinates(t, X_t), with
    the weights (paths, nodes - 1), by default equal.
    """
    times = jnp.asarray(times, dtype=float)
    simulated = simulate(heston, times, stream, paths)
    x = simulated.log_spot(_MeanLeverage(heston))[:, :-1]
    v = (simulated.variance * _mean(heston, times)[None, :])[:, :-1]
    t = jnp.broadcast_to(times[:-1], x.shape)
    weights = jnp.ones(x.shape) if weights is None else jnp.asarray(weights, dtype=float)
    z = coordinates(t, x).reshape(-1, 2)
    alpha = regression.fit(z, v.ravel(), weights.ravel())
    return HestonProjection(compiled(coordinates), regression.basis, alpha)
