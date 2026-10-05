"""The parametrized leverage factor

    L_beta(t, x) = L_0(t, x) exp(b(tau, u)^T beta),

with a base L_0, for instance the capped Dupire local volatility, the coordinates (tau, u) of a
chart and a basis b on [-1, 1]^2.
"""

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp

from locsvol.functions import compiled
from locsvol.leverage.chart import Chart
from locsvol.regression.chebyshev import ChebyshevTensor, chebyshev


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["base", "chart", "coefficients"],
    meta_fields=["basis"],
)
@dataclass(frozen=True)
class Parametrized:
    """L_beta = L_0 exp(b(chart(t, x))^T beta)."""

    base: object
    chart: Chart
    basis: object
    coefficients: jax.Array

    def __post_init__(self):
        object.__setattr__(self, "base", compiled(self.base))

    def coordinates(self, t, x):
        return self.chart(t, x)

    def log_factor(self, t, x):
        """b(chart(t, x))^T beta; for the Chebyshev tensor basis
        sum_j T_j(u) sum_i T_i(tau) beta_ij."""
        z = self.chart(t, x)
        if not isinstance(self.basis, ChebyshevTensor):
            return self.basis(z) @ self.coefficients
        p, q = self.basis.degrees
        table = self.coefficients.reshape(p + 1, q + 1)
        state = chebyshev(z[..., 1], q)
        if jnp.ndim(t) == 0:
            time = chebyshev(self.chart(t, jnp.asarray(0.0))[0], p)
            values = state @ (time @ table)
        else:
            values = jnp.sum((chebyshev(z[..., 0], p) @ table) * state, axis=-1)
        return jnp.where(jnp.isfinite(z[..., 0]), values, jnp.nan)

    def __call__(self, t, x):
        value = jnp.exp(jnp.log(self.base(t, x)) + self.log_factor(t, x))
        t, x = jnp.broadcast_arrays(jnp.asarray(t), jnp.asarray(x))
        valid = jnp.isfinite(t) & (t >= 0) & jnp.isfinite(x) & jnp.isfinite(value) & (value > 0)
        return jnp.where(valid, value, jnp.nan)

    def shifted(self, delta):
        """L_{beta + delta}."""
        return replace(self, coefficients=self.coefficients + delta)
