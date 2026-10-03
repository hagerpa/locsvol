"""Tensor products of Chebyshev polynomials on [-1, 1]^2.

The basis of degrees (p, q) consists of the (p + 1)(q + 1) functions

    b_{i(q+1)+j}(u, v) = T_i(u) T_j(v),    0 <= i <= p,  0 <= j <= q,

so the first coordinate (time) is the outer index. Since 2 T_a T_b = T_{a+b} + T_{|a-b|}, the Gram
matrix sum_n w_n b(u_n, v_n) b(u_n, v_n)^T follows from the (2p + 1)(2q + 1) moments
sum_n w_n T_a(u_n) T_b(v_n), which is much cheaper than summing outer products.
"""

from dataclasses import dataclass

import jax.numpy as jnp


def chebyshev(x, degree):
    """T_0(x), ..., T_degree(x) by T_{k+1} = 2x T_k - T_{k-1}, stacked along a new last axis."""
    columns = [jnp.ones_like(x)]
    if degree > 0:
        columns.append(x)
    for _ in range(degree - 1):
        columns.append(2 * x * columns[-1] - columns[-2])
    return jnp.stack(columns, axis=-1)


@dataclass(frozen=True)
class ChebyshevTensor:
    """Chebyshev tensor basis of degrees (p, q) in the time and the state coordinate."""

    degrees: tuple[int, int]

    @property
    def size(self):
        p, q = self.degrees
        return (p + 1) * (q + 1)

    def __call__(self, x):
        """The basis functions at coordinates x of shape (..., 2), shape (..., size)."""
        p, q = self.degrees
        u, v = chebyshev(x[..., 0], p), chebyshev(x[..., 1], q)
        return (u[..., :, None] * v[..., None, :]).reshape(x.shape[:-1] + (self.size,))

    def moments(self, x, weights, values):
        """The moments M[a, b] = sum_n w_n T_a(u_n) T_b(v_n), a <= 2p, b <= 2q, and the projection
        sum_n w_n z_n b(x_n) of the values z_n."""
        p, q = self.degrees
        u, v = chebyshev(x[:, 0], 2 * p), chebyshev(x[:, 1], 2 * q)
        moments = jnp.einsum("ca,cb,c->ab", u, v, weights)
        projection = jnp.einsum("ca,cb,c->ab", u[:, : p + 1], v[:, : q + 1], weights * values)
        return moments, projection.reshape(self.size)

    def gram(self, moments):
        """sum_n w_n b(x_n) b(x_n)^T from the moments of moments()."""
        p, q = self.degrees
        i, j = jnp.arange(p + 1), jnp.arange(q + 1)
        time = (i[:, None] + i, jnp.abs(i[:, None] - i))
        state = (j[:, None] + j, jnp.abs(j[:, None] - j))
        quarters = jnp.stack(
            [moments[a[:, :, None, None], b[None, None, :, :]] for a in time for b in state]
        ).sum(axis=0)
        return 0.25 * quarters.transpose(0, 2, 1, 3).reshape(self.size, self.size)
