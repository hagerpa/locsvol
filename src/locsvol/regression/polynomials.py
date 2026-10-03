"""Polynomial bases: monomials, Chebyshev polynomials of one coordinate, and the identity basis.

The power basis with exponent vectors p_1, ..., p_B in d coordinates consists of the monomials

    b_j(x) = x^{p_j} = x_1^{p_j1} ... x_d^{p_jd}.

Two variants keep the features bounded far from the origin: each monomial is clipped to an
interval [lo, hi], or the basis is made rational,

    b_j(x) = x^{p_j} / (1 + |x_1|^r + ... + |x_d|^r),    r >= |p_j| = p_j1 + ... + p_jd,

so that every feature is bounded. The rational features are evaluated in the scaled variables
x / s with s = max(1, |x_1|, ..., |x_d|),

    b_j(x) = (x / s)^{p_j} s^{|p_j| - r} / (s^{-r} + |x_1 / s|^r + ... + |x_d / s|^r),

which avoids overflow of the powers.

The Chebyshev basis of degree q in one coordinate u, for instance the state coordinate of a chart
without time, consists of T_0(u), ..., T_q(u); see chebyshev.py for the tensor products in time
and state. The identity basis b(x) = x takes the coordinates themselves as features, for instance
the columns of a design matrix built by the caller.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from locsvol.regression.chebyshev import chebyshev


@dataclass(frozen=True)
class Identity:
    """b(x) = x for coordinates of the given width."""

    width: int

    @property
    def size(self):
        return self.width

    def __call__(self, x):
        """The coordinates x of shape (..., width) as features."""
        return jnp.asarray(x)


@dataclass(frozen=True)
class Powers:
    """Monomials x^{p_j}, optionally rational with degree r or clipped to bounds (lo, hi)."""

    powers: tuple[tuple[int, ...], ...]
    rational_degree: int | None = None
    bounds: tuple[float, float] | None = None

    def __post_init__(self):
        if self.rational_degree is not None and self.bounds is not None:
            raise ValueError("a power basis is either rational or clipped")
        if self.rational_degree is not None and self.rational_degree < max(map(sum, self.powers)):
            raise ValueError("the rational degree must be at least the largest total degree")

    @property
    def size(self):
        return len(self.powers)

    def __call__(self, x):
        """The monomials at coordinates x of shape (..., d), shape (..., size)."""
        x = jnp.asarray(x)
        powers = jnp.asarray(self.powers, dtype=jnp.int32)
        if self.rational_degree is not None:
            r = self.rational_degree
            scale = jnp.maximum(1.0, jnp.max(jnp.abs(x), axis=-1))
            reduced = x / scale[..., None]
            terms = jnp.prod(reduced[..., None, :] ** powers, axis=-1)
            denominator = scale ** (-r) + jnp.sum(jnp.abs(reduced) ** r, axis=-1)
            return (
                terms * scale[..., None] ** (jnp.sum(powers, axis=-1) - r) / denominator[..., None]
            )
        terms = jnp.prod(x[..., None, :] ** powers, axis=-1)
        return terms if self.bounds is None else jnp.clip(terms, *self.bounds)


@dataclass(frozen=True)
class Chebyshev:
    """T_0(u), ..., T_q(u) of one coordinate u."""

    degree: int

    @property
    def size(self):
        return self.degree + 1

    def __call__(self, x):
        """The polynomials at coordinates x of shape (..., 1), shape (..., degree + 1)."""
        return chebyshev(jnp.asarray(x)[..., 0], self.degree)
