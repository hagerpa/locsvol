"""Tensor products of B-splines in time and Chebyshev polynomials in the state.

On the time domain [a, b], a clamped knot vector k_0 = ... = k_p = a < k_{p+1} <= ... <= k_{K-p-2}
< k_{K-p-1} = ... = k_{K-1} = b of degree p gives the B-splines B_{0,p}, ..., B_{K-p-2,p} (see
splines.py), and the basis of state degree q consists of

    b_{i(q+1)+j}(tau, u) = B_{i,p}(tau) T_j(u),    0 <= i < K - p - 1,  0 <= j <= q,

with the time function as the outer index. An interior knot of multiplicity m <= max(1, p) makes
the B-splines p - m times continuously differentiable there; cubic B-splines with simple interior
knots are twice continuously differentiable in time.

Beyond [a, b] the time functions keep their end values ('constant'), follow their tangents
('linear') or are undefined ('reject', nan). Beyond the state domain [c, d] the polynomials are
evaluated as they are ('polynomial') or are undefined ('reject'). Since the B-splines sum to one
on [a, b], the functions B_{i,p}(tau) T_0(u) represent the constant 1 with all coefficients one.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from locsvol.regression.chebyshev import chebyshev
from locsvol.regression.splines import spline_columns


@dataclass(frozen=True)
class TimeSplineChebyshev:
    """B-splines of degree time_degree on clamped knots in time times Chebyshev polynomials up to
    state_degree in the state; see the module docstring."""

    knots: tuple[float, ...]
    time_degree: int
    state_degree: int
    time_domain: tuple[float, float] = (-1.0, 1.0)
    state_domain: tuple[float, float] = (-1.0, 1.0)
    time_extrapolation: str = "constant"
    state_extrapolation: str = "polynomial"

    def __post_init__(self):
        if self.time_extrapolation not in ("constant", "linear", "reject"):
            raise ValueError("time extrapolation must be 'constant', 'linear' or 'reject'")
        if self.state_extrapolation not in ("polynomial", "reject"):
            raise ValueError("state extrapolation must be 'polynomial' or 'reject'")
        k, end = np.asarray(self.knots, dtype=float), self.time_degree + 1
        lower, upper = self.time_domain
        if (
            k.ndim != 1
            or k.size < 2 * end
            or not np.all(np.isfinite(k))
            or not np.all(np.diff(k) >= 0)
            or not np.all(k[:end] == lower)
            or not np.all(k[-end:] == upper)
        ):
            raise ValueError("the time knots must be ordered and clamped to the time domain")
        interior = k[end:-end]
        _, multiplicities = np.unique(interior, return_counts=True)
        if np.any((interior <= lower) | (interior >= upper)) or np.any(
            multiplicities > max(1, self.time_degree)
        ):
            raise ValueError("interior time knots must lie inside the domain, repeated <= degree")

    @property
    def sizes(self):
        """The numbers of time and state functions."""
        return len(self.knots) - self.time_degree - 1, self.state_degree + 1

    @property
    def size(self):
        time, state = self.sizes
        return time * state

    @property
    def unit(self):
        """The coefficients of the constant function 1: one at every T_0 term, else zero."""
        return jnp.zeros(self.sizes).at[:, 0].set(1.0).reshape(-1)

    def __call__(self, x):
        """The basis functions at coordinates (tau, u) of shape (..., 2), shape (..., size)."""
        x = jnp.asarray(x, dtype=jnp.float64)
        tau, u = x[..., 0], x[..., 1]
        knots = jnp.asarray(self.knots, dtype=jnp.float64)
        time = spline_columns(tau, knots, self.time_degree, self.time_extrapolation)
        state = chebyshev(u, self.state_degree)
        valid_time, valid_state = jnp.isfinite(tau), jnp.isfinite(u)
        if self.time_extrapolation == "reject":
            valid_time &= (tau >= self.time_domain[0]) & (tau <= self.time_domain[1])
        if self.state_extrapolation == "reject":
            valid_state &= (u >= self.state_domain[0]) & (u <= self.state_domain[1])
        time = jnp.where(valid_time[..., None], time, jnp.nan)
        state = jnp.where(valid_state[..., None], state, jnp.nan)
        return (time[..., :, None] * state[..., None, :]).reshape(x.shape[:-1] + (self.size,))
