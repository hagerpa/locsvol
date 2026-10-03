"""B-spline bases and the Gamma regression on B-splines with a difference penalty.

On a knot vector k_0 < k_1 < ... < k_{K-1}, the K - p - 1 B-splines of degree p follow from the
recursion of Cox and de Boor (de Boor, A Practical Guide to Splines, 1978, Ch. IX),

    B_{i,0}(x) = 1{k_i <= x < k_{i+1}},
    B_{i,p}(x) = (x - k_i) / (k_{i+p} - k_i) B_{i,p-1}(x)
                 + (k_{i+p+1} - x) / (k_{i+p+1} - k_{i+1}) B_{i+1,p-1}(x).

On the base interval [k_p, k_{K-p-1}] they sum to one. Beyond it the basis is extended by its
values at the nearest end ('constant') or by its tangent there ('linear'). Several coordinates
give the tensor products of their B-splines, the first coordinate as the outer index; a
coordinate with an empty knot vector contributes the constant function 1.

The knots are placed by the observations: for each coordinate, n base knots at the linear
quantiles of the observations at the probabilities 0, 1/(n-1), ..., 1, without repetitions
('quantile'), or n equally spaced points of [-1, 1] ('uniform'); p further knots on each side
continue the spacing of the outer base intervals. A coordinate whose observations are constant
(up to rounding) gets the constant function.

The Gamma regression on these B-splines penalizes second differences of the coefficients,
lambda |D alpha|^2 / 2 in each coordinate (the P-splines of Eilers and Marx, Statist. Sci. 11,
1996), that is alpha^T P alpha / 2 with

    P = sum_i lambda_i I x ... x D_i^T D_i x ... x I,

D_i the second-difference matrix of coordinate i and x the Kronecker product. It starts from the
constant mean of the observations, which B-splines represent by equal coefficients.
"""

from dataclasses import dataclass
from functools import partial
from math import prod

import jax
import jax.numpy as jnp

from locsvol.regression.gamma import GammaRegression
from locsvol.regression.means import LogLinear

EXTRAPOLATIONS = ("constant", "linear")


def bsplines(x, knots, degree):
    """B_{0,p}(x), ..., B_{K-p-2,p}(x) of the knots, stacked along a new last axis.

    Every x is placed in a knot interval [k_i, k_{i+1}) with p <= i <= K - p - 2, so the last
    base interval is closed on the right.
    """
    index = jnp.arange(knots.size - 1)
    interval = jnp.clip(
        jnp.searchsorted(knots, x, side="right") - 1, degree, knots.size - degree - 2
    )

    def raise_degree(order, values):
        end = jnp.minimum(index + order, knots.size - 1)
        following = jnp.minimum(index + order + 1, knots.size - 1)
        left_width = knots[end] - knots[index]
        right_width = knots[following] - knots[index + 1]
        left = (x[..., None] - knots[index]) / jnp.where(left_width > 0, left_width, 1.0)
        right = (knots[following] - x[..., None]) / jnp.where(right_width > 0, right_width, 1.0)
        shifted = jnp.concatenate((values[..., 1:], jnp.zeros_like(values[..., :1])), axis=-1)
        return jnp.where(index + order + 1 < knots.size, left * values + right * shifted, 0.0)

    values = (interval[..., None] == index).astype(x.dtype)
    values = jax.lax.fori_loop(1, degree + 1, raise_degree, values)
    return values[..., : knots.size - degree - 1]


def spline_columns(x, knots, degree, extrapolation):
    """The B-splines of one coordinate with their extension beyond the base interval."""
    if knots.size == 0:
        return jnp.ones(x.shape + (1,), dtype=x.dtype)
    lower, upper = knots[degree], knots[-degree - 1]
    values = bsplines(jnp.where(x < lower, lower, jnp.where(x > upper, upper, x)), knots, degree)
    if extrapolation == "linear":
        ends = jnp.stack((lower, upper))
        _, slopes = jax.jvp(partial(bsplines, knots=knots, degree=degree), (ends,), (jnp.ones(2),))
        values = values + jnp.where(x < lower, x - lower, 0.0)[..., None] * slopes[0]
        values = values + jnp.where(x > upper, x - upper, 0.0)[..., None] * slopes[1]
    return values


def tensor(margins):
    """Products of the margins' functions, the first margin as the outer index."""
    values = margins[0]
    for margin in margins[1:]:
        values = values[..., :, None] * margin[..., None, :]
        values = values.reshape(values.shape[:-2] + (-1,))
    return values


@dataclass(frozen=True)
class Splines:
    """Tensor products of B-splines of one degree, one knot vector per coordinate.

    The knots are tuples of floats, so that the basis is a static argument of compiled code.
    """

    knots: tuple[tuple[float, ...], ...]
    degree: int = 3
    extrapolation: str = "constant"

    def __post_init__(self):
        if self.extrapolation not in EXTRAPOLATIONS:
            raise ValueError(f"extrapolation must be one of {EXTRAPOLATIONS}")
        if any(0 < len(k) < 2 * self.degree + 2 for k in self.knots):
            raise ValueError("a knot vector needs at least 2 (degree + 1) knots")

    @property
    def sizes(self):
        """The number of functions of each coordinate."""
        return tuple(1 if not k else len(k) - self.degree - 1 for k in self.knots)

    @property
    def size(self):
        return prod(self.sizes)

    @property
    def unit(self):
        """The coefficients of the constant function 1: all ones, by the partition of unity."""
        return jnp.ones(self.size)

    def __call__(self, x):
        """The basis functions at coordinates x of shape (..., d), shape (..., size)."""
        x = jnp.asarray(x)
        knots = [jnp.asarray(k, dtype=jnp.float64) for k in self.knots]
        return tensor(
            [
                spline_columns(x[..., i], k, self.degree, self.extrapolation)
                for i, k in enumerate(knots)
            ]
        )


def spline_basis(x, counts, degree=3, rule="quantile", extrapolation="constant"):
    """The B-splines with knots placed by the observations x of shape (n, d), with counts[i]
    base knots in coordinate i; see the module docstring."""
    if rule not in ("quantile", "uniform"):
        raise ValueError("the knot rule must be 'quantile' or 'uniform'")
    x = jnp.asarray(x, dtype=jnp.float64)
    if x.ndim != 2 or x.shape[1] != len(counts) or not bool(jnp.all(jnp.isfinite(x))):
        raise ValueError("observations must be finite of shape (n, len(counts))")
    knots = []
    for column, count in zip(x.T, counts, strict=True):
        if rule == "uniform":
            base = jnp.linspace(-1.0, 1.0, count)
        else:
            scale = jnp.maximum(1.0, jnp.max(jnp.abs(column)))
            if bool(jnp.ptp(column) <= jnp.finfo(x.dtype).eps * scale):
                knots.append(())
                continue
            base = jnp.unique(jnp.quantile(column, jnp.linspace(0.0, 1.0, count), method="linear"))
        if base.size < 2:
            knots.append(())
            continue
        first, last = base[1] - base[0], base[-1] - base[-2]
        full = jnp.concatenate(
            (
                jnp.linspace(base[0] - degree * first, base[0] - first, degree),
                base,
                jnp.linspace(base[-1] + last, base[-1] + degree * last, degree),
            )
        )
        if not bool(jnp.all(jnp.isfinite(full)) & jnp.all(jnp.diff(full) > 0)):
            raise ValueError("the knots must be finite and strictly increasing")
        knots.append(tuple(float(k) for k in full))
    return Splines(tuple(knots), degree, extrapolation)


def spline_penalty(sizes, weights):
    """P = sum_i lambda_i I x ... x D_i^T D_i x ... x I for coordinates with sizes[i] functions;
    a coordinate with fewer than three functions has no second differences."""
    total = None
    for i, weight in enumerate(weights):
        differences = jnp.diff(jnp.eye(sizes[i]), n=2, axis=0)
        factors = [jnp.eye(n) for n in sizes]
        factors[i] = differences.T @ differences
        term = factors[0]
        for factor in factors[1:]:
            term = jnp.kron(term, factor)
        total = weight * term if total is None else total + weight * term
    return total


@dataclass(frozen=True)
class GammaSplines:
    """Gamma regression with log link on B-splines placed by the observations, with the
    second-difference penalties lambda_i of the coordinates; see the module docstring.

    initialization is 'partition' (the constant mean of the observations) or 'zeros'.
    """

    counts: tuple[int, ...]
    penalties: tuple[float, ...]
    degree: int = 3
    rule: str = "quantile"
    extrapolation: str = "constant"
    initialization: str = "partition"
    tolerance: float = 1e-8
    max_iterations: int = 300
    max_evaluations: int = 600
    block_rows: int = 32768

    def fit(self, x, y, weights, offset=0.0):
        """The fitted mean exp(o + b(x)^T alpha), with knots placed by all rows of x, including
        rows of weight zero."""
        basis = spline_basis(x, self.counts, self.degree, self.rule, self.extrapolation)
        regression = GammaRegression(
            basis,
            spline_penalty(basis.sizes, self.penalties),
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            max_evaluations=self.max_evaluations,
            block_rows=self.block_rows,
            initialization=self.initialization,
        )
        return LogLinear(basis, regression.fit(x, y, weights, offset))
