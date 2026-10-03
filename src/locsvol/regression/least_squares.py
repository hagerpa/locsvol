"""Weighted least squares by the truncated singular value decomposition.

The coefficients minimize

    sum_n w_n (y_n - b(x_n)^T alpha)^2 / sum_n w_n

over alpha. With the weighted design A = diag(sqrt(w / sum w)) B, B_nj = b_j(x_n), and its thin
singular value decomposition A = U S V^T, the singular values s_i <= rcond s_1 are dropped and

    alpha = V S^+ U^T diag(sqrt(w / sum w)) y,

the solution of least norm in the span of the kept singular vectors (Golub and Van Loan, Matrix
Computations, 4th ed., 2013, Sec. 5.5). The fitted mean b(x)^T alpha is linear, without link,
offset or positivity, and serves to compare with the regressions with log link.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames="basis")
def _solve(x, y, weights, rcond, basis):
    scale = jnp.sqrt(weights / jnp.sum(weights))
    u, singular, vt = jnp.linalg.svd(scale[:, None] * basis(x), full_matrices=False)
    keep = singular > rcond * singular[0]
    return vt.T @ jnp.where(keep, (u.T @ (scale * y)) / jnp.where(keep, singular, 1.0), 0.0)


@dataclass(frozen=True)
class LeastSquares:
    """Weighted least squares on a basis with relative cutoff rcond of the singular values."""

    basis: object
    rcond: float

    def fit(self, x, y, weights):
        """The coefficients alpha for coordinates x (n, d), responses y and weights w >= 0."""
        x, y, w = (jnp.asarray(v, dtype=jnp.float64) for v in (x, y, weights))
        if not bool(jnp.all(w >= 0) & (jnp.sum(w) > 0)):
            raise ValueError("weights must be nonnegative with a positive sum")
        return _solve(x, y, w, self.rcond, self.basis)
