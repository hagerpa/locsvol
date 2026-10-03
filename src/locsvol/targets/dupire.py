"""Dupire's local volatility of a total variance surface (Dupire, Pricing with a smile, Risk,
1994; Gatheral, The Volatility Surface, Wiley, 2006, Ch. 1).

For a total variance surface w(t, k) without arbitrage, the local volatility model
dX_t = -sigma^2/2 dt + sigma(t, X_t) dW_t for X = log(S/F) reproduces its option prices when

    sigma_Dup(t, x)^2 = d_T w / g,
    g = 1 - x/w d_k w + 1/4 (-1/4 - 1/w + x^2/w^2) (d_k w)^2 + 1/2 d_kk w,

with w and its derivatives at (t, x). The density factor g, evaluated as
(1 - x w_k / (2w))^2 - w_k^2/4 (1/w + 1/4) + w_kk/2, is positive exactly where the smile is free
of butterfly arbitrage (Gatheral and Jacquier, Quant. Finance 14, 2014, Lemma 2.2). The local
variance is nan wherever w, d_T w, g or their ratio is not positive and finite.

At t = 0 the formula is replaced by its limit as t -> 0 where the surface has one: at x = 0 for
SSVI and for a self-similar short extension of SVI, and for every x for a shifted one.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


def butterfly_factor(w, k, w_k, w_kk):
    """The density factor g of the smile w at k."""
    root = jnp.sqrt(w)
    scaled = w_k / root
    slope = 0.5 * k * (scaled / root)
    return (
        jnp.square(1.0 - slope) - (jnp.square(0.5 * scaled) + jnp.square(0.25 * w_k)) + 0.5 * w_kk
    )


def local_variance(w, w_t, w_k, w_kk, k):
    """w_t / g, nan unless w, w_t, g and the ratio are positive and finite."""
    g = butterfly_factor(w, k, w_k, w_kk)
    ratio = w_t / g
    valid = (
        jnp.isfinite(w)
        & (w > 0.0)
        & jnp.isfinite(w_t)
        & (w_t > 0.0)
        & jnp.isfinite(w_k)
        & jnp.isfinite(w_kk)
        & jnp.isfinite(g)
        & (g > 0.0)
        & jnp.isfinite(ratio)
        & (ratio > 0.0)
    )
    return jnp.where(valid, ratio, jnp.nan)


@jax.jit
def _local_variance(surface, t, x):
    t, x = jnp.broadcast_arrays(jnp.asarray(t, dtype=float), jnp.asarray(x, dtype=float))
    positive = local_variance(*surface.jet(t, x), x)
    limit = surface.initial_local_variance(x)
    limit = jnp.where(jnp.isfinite(limit) & (limit > 0.0), limit, jnp.nan)
    return jnp.where(t == 0.0, limit, positive)


@partial(jax.tree_util.register_dataclass, data_fields=["surface"], meta_fields=[])
@dataclass(frozen=True)
class Dupire:
    """sigma_Dup(t, x) of a total variance surface, such as SSVI or RefinedSVI."""

    surface: object

    @property
    def log_moneyness_bounds(self):
        """The interval of x on which sigma_Dup is defined."""
        return self.surface.log_moneyness_bounds

    def local_variance(self, t, x):
        """sigma_Dup(t, x)^2 at broadcast t and x = log(S/F), nan outside its domain."""
        return _local_variance(self.surface, t, x)

    def __call__(self, t, x):
        """sigma_Dup(t, x) at broadcast t and x = log(S/F), nan outside its domain."""
        return jnp.sqrt(self.local_variance(t, x))
