"""Nadaraya-Watson kernel regression.

The conditional mean of y given s is estimated at a query point q by

    m(q) = sum_k K((s_k - q) / h) y_k / sum_k K((s_k - q) / h),

with the quartic kernel K(u) = (15/16)(1 - u^2)^2 on |u| < 1, as in the particle method of Guyon
and Henry-Labordere, or the Gaussian kernel. The constant factor of K cancels. A query without an
observation inside the quartic window has no estimate (nan). The sums run over blocks of
observations and of queries, so memory stays bounded.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


@partial(jax.jit, static_argnames=("kernel", "block_rows", "query_rows"))
def kernel_means(s, y, queries, bandwidth, kernel="quartic", block_rows=4096, query_rows=256):
    """m(q) at the queries for observations (s_k, y_k) and the bandwidth h."""
    n = s.size
    size = min(n, block_rows)
    pad = (-n) % size
    blocks = (
        jnp.pad(s, (0, pad)).reshape(-1, size),
        jnp.pad(y, (0, pad), constant_values=1.0).reshape(-1, size),
        (jnp.arange(n + pad) < n).reshape(-1, size),
    )
    inputs_valid = (
        jnp.all(jnp.isfinite(s) & (s >= 0))
        & jnp.all(jnp.isfinite(y) & (y > 0))
        & jnp.isfinite(bandwidth)
        & (bandwidth > 0)
    )

    def gaussian(query):
        def add(carry, block):
            maximum, mass, numerator = carry
            levels, response, active = block
            logs = jnp.where(active, -0.5 * ((levels - query) / bandwidth) ** 2, -jnp.inf)
            new_maximum = jnp.maximum(maximum, jnp.max(logs))
            safe = jnp.where(jnp.isfinite(new_maximum), new_maximum, 0.0)
            scale = jnp.where(jnp.isfinite(maximum), jnp.exp(maximum - safe), 0.0)
            weights = jnp.where(active, jnp.exp(logs - safe), 0.0)
            return (
                new_maximum,
                scale * mass + jnp.sum(weights),
                scale * numerator + jnp.sum(weights * response),
            ), None

        start = (jnp.asarray(-jnp.inf), jnp.array(0.0), jnp.array(0.0))
        (_, mass, numerator), _ = lax.scan(add, start, blocks)
        return mass, numerator

    def quartic(query):
        def add(carry, block):
            mass, numerator = carry
            levels, response, active = block
            u = (levels - query) / bandwidth
            weights = jnp.where(active & (jnp.abs(u) < 1), (1 - u**2) ** 2, 0.0)
            return (mass + jnp.sum(weights), numerator + jnp.sum(weights * response)), None

        (mass, numerator), _ = lax.scan(add, (jnp.array(0.0), jnp.array(0.0)), blocks)
        return mass, numerator

    def estimate(query):
        mass, numerator = (gaussian if kernel == "gaussian" else quartic)(query)
        mean = numerator / mass
        valid = (
            inputs_valid
            & jnp.isfinite(query)
            & (query > 0)
            & jnp.isfinite(mean)
            & (mean > 0)
            & jnp.isfinite(mass)
            & (mass > 0)
        )
        return jnp.where(valid, mean, jnp.nan)

    count = queries.size
    block = min(count, query_rows)
    padded = jnp.pad(queries, (0, (-count) % block), constant_values=jnp.nan)
    means = lax.map(jax.vmap(estimate), padded.reshape(-1, block))
    return means.reshape(-1)[:count]
