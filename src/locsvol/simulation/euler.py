"""The Euler scheme of the local stochastic volatility model.

With X = log(S/F) and the variance factor Y of the backbone, the model

    dX_t = -L(t, S_t)^2 Y_t dt / 2 + L(t, S_t) sqrt(Y_t) dW_t

is simulated by the Euler scheme with left-point coefficients,

    X_{j+1} = X_j - sigma_j^2 (t_{j+1} - t_j) / 2 + sigma_j (W_{t_{j+1}} - W_{t_j}),
    sigma_j = L(t_j, X_j) sqrt(Y_{t_j}),   X_0 = 0.

The variance factor does not depend on the leverage factor, so its paths are simulated once and
reused for every leverage factor.
"""

import jax
import jax.numpy as jnp

from locsvol.functions import compiled


def euler_step(leverage, t, dt, x, y, dw):
    """X_{j+1} from X_j = x, Y_{t_j} = y and the increment dw; nan where a value is invalid."""
    valid = jnp.isfinite(y) & (y >= 0)
    sigma = leverage(t, x) * jnp.sqrt(jnp.where(valid, y, 0.0))
    following = x - 0.5 * sigma**2 * dt + sigma * dw
    valid &= jnp.isfinite(x) & jnp.isfinite(sigma) & (sigma >= 0) & jnp.isfinite(following)
    return jnp.where(valid, following, jnp.nan)


def log_spot(leverage, times, variance, spot_increments, start=None):
    """Paths of X = log(S/F) at all nodes, shape (paths, nodes).

    variance holds Y at the nodes (paths, nodes) and spot_increments the increments of W
    (paths, nodes - 1). The paths start at X = start at the first node, or at X = 0. A path
    becomes nan from the first step with an invalid value.
    """
    x0 = jnp.zeros(variance.shape[0]) if start is None else jnp.asarray(start)
    return _log_spot(compiled(leverage), times, variance, spot_increments, x0)


@jax.jit
def _log_spot(leverage, times, variance, spot_increments, x0):

    def step(x, j):
        following = euler_step(
            leverage,
            times[j],
            times[j + 1] - times[j],
            x,
            variance[:, j],
            spot_increments[:, j],
        )
        return following, following

    _, values = jax.lax.scan(step, x0, jnp.arange(times.size - 1, dtype=jnp.int32))
    return jnp.concatenate((x0[:, None], values.T), axis=1)
