"""Bases and regression methods of the caller.

A basis is a function b mapping coordinates x of shape (..., d) to features of shape (..., B);
any such JAX function serves as the basis of a Gamma regression, of the parametrized leverage
factor or of the time-interpolated coefficients. The fit evaluates the basis at the origin for
padding rows, so b(0) must be finite.

A regression method fits the conditional mean of a response y given coordinates x from weighted
observations. A method of the caller is a function fit(x, y, weights) returning the fitted mean,
a positive JAX function m of coordinates z of shape (..., d) with values of shape (...); it can
replace the built-in methods in the progressive calibration and in per_time_means.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CustomBasis:
    """The basis b(x) = function(x) with size features."""

    function: object
    size: int

    def __call__(self, x):
        """The features at coordinates x of shape (..., d), shape (..., size)."""
        values = self.function(x)
        if values.shape[-1] != self.size:
            raise ValueError(f"the basis returns {values.shape[-1]} features, not {self.size}")
        return values


@dataclass(frozen=True)
class CustomRegression:
    """The regression method fit(x, y, weights) -> fitted mean m(z)."""

    function: object

    def fit(self, x, y, weights):
        """The fitted mean of the method on coordinates x (n, d), responses y and weights."""
        return self.function(x, y, weights)
