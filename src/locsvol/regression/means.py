"""Fitted conditional means and means that are piecewise constant in time.

A regression with log link fits the mean

    m(z, o) = exp(o + b(z)^T alpha)

with an offset o. Fitting a regression separately at the nodes t_0 < ... < t_n of a time grid
gives means m_0, ..., m_{n-1} and the mean that is piecewise constant in time,

    m(t, z) = m_j(z)   for t_j <= t < t_{j+1},   and m(t_n, z) = m_{n-1}(z).

Means of equal structure (for instance coefficients on one basis, or tables of equal capacity)
are stacked along a leading axis and selected by index; others are selected by a switch.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.functions import compiled
from locsvol.regression.gamma import GammaRegression


@partial(jax.tree_util.register_dataclass, data_fields=["coefficients"], meta_fields=["basis"])
@dataclass(frozen=True)
class LogLinear:
    """m(z, o) = exp(o + b(z)^T alpha); nan where a coordinate or the mean is not finite."""

    basis: object
    coefficients: jax.Array

    def __call__(self, z, offset=0.0):
        z = jnp.asarray(z)
        eta = self.basis(z) @ self.coefficients + offset
        mean = jnp.exp(eta)
        valid = jnp.all(jnp.isfinite(z), axis=-1) & jnp.isfinite(eta) & (mean > 0)
        return jnp.where(valid & jnp.isfinite(mean), mean, jnp.nan)


def fitted_mean(regression, x, y, weights):
    """The conditional mean fitted by the regression: for a GammaRegression the log-linear mean
    of its coefficients, otherwise the fitted mean the regression returns."""
    fitted = regression.fit(x, y, weights)
    return (
        LogLinear(regression.basis, fitted) if isinstance(regression, GammaRegression) else fitted
    )


@partial(jax.tree_util.register_dataclass, data_fields=["times", "means"], meta_fields=["stacked"])
@dataclass(frozen=True)
class PerTime:
    """m(t, z) = m_j(z) for t_j <= t < t_{j+1}; nan for t outside [t_0, t_n].

    means holds m_0, ..., m_{n-1}: stacked along a leading axis when stacked is True, otherwise
    a tuple.
    """

    times: jax.Array
    means: object
    stacked: bool

    def index(self, t):
        """j with t_j <= t < t_{j+1}, and n - 1 at t_n."""
        return jnp.clip(jnp.searchsorted(self.times, t, side="right") - 1, 0, self.times.size - 2)

    def _at(self, j, z):
        if self.stacked:
            return jax.tree.map(lambda a: a[j], self.means)(z)
        branches = [lambda z, mean=mean: mean(z) for mean in self.means]
        return jax.lax.switch(j, branches, z)

    def __call__(self, t, z):
        """The mean at times t and coordinates z of shape (..., d); t broadcasts to z[..., 0]."""
        t, z = jnp.asarray(t), jnp.asarray(z)
        if t.ndim == 0:
            values = self._at(self.index(t), z)
        else:
            t = jnp.broadcast_to(t, z.shape[:-1])
            queries = (t.reshape(-1), z.reshape(-1, z.shape[-1]))

            def single(query):
                return self._at(self.index(query[0]), query[1][None])[0]

            # A switch under vmap would evaluate every mean at every query.
            values = jax.vmap(single)(queries) if self.stacked else jax.lax.map(single, queries)
            values = values.reshape(t.shape)
        valid = jnp.isfinite(t) & (t >= self.times[0]) & (t <= self.times[-1])
        return jnp.where(valid, values, jnp.nan)


def per_time(times, means):
    """The piecewise constant mean of the means m_0, ..., m_{n-1} on the grid times."""
    times = jnp.asarray(times, dtype=jnp.float64)
    if len(means) != times.size - 1:
        raise ValueError("one mean per interval of the time grid is required")
    means = [compiled(mean) for mean in means]
    structures = {jax.tree.structure(mean) for mean in means}
    shapes = {tuple(jnp.shape(a) for a in jax.tree.leaves(mean)) for mean in means}
    if len(structures) == 1 and len(shapes) == 1:
        return PerTime(times, jax.tree.map(lambda *a: jnp.stack(a), *means), True)
    return PerTime(times, tuple(means), False)


def per_time_means(regression, times, x, y, weights):
    """The regression fitted separately at the nodes t_0, ..., t_{n-1} of the grid times, from
    coordinates x (paths, n, d), responses y (paths, n) and weights (paths, n)."""
    means = [fitted_mean(regression, x[:, j], y[:, j], weights[:, j]) for j in range(x.shape[1])]
    return per_time(times, means)
