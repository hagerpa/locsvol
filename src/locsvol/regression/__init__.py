"""Regression methods for the conditional mean of the variance factor given the spot, and the bases
they use."""

from locsvol.regression.bins import Bins, QuantileBins
from locsvol.regression.chebyshev import ChebyshevTensor
from locsvol.regression.cubic import flat_cubic, not_a_knot
from locsvol.regression.custom import CustomBasis, CustomRegression
from locsvol.regression.gamma import GammaRegression
from locsvol.regression.kernel import kernel_means
from locsvol.regression.least_squares import LeastSquares
from locsvol.regression.means import LogLinear, PerTime, fitted_mean, per_time, per_time_means
from locsvol.regression.mixed import TimeSplineChebyshev
from locsvol.regression.network import Network, PositiveMeanNetwork
from locsvol.regression.polynomials import Chebyshev, Identity, Powers
from locsvol.regression.splines import GammaSplines, Splines, spline_basis, spline_penalty
from locsvol.regression.tweedie import tweedie_loss

__all__ = [
    "Bins",
    "Chebyshev",
    "ChebyshevTensor",
    "CustomBasis",
    "CustomRegression",
    "GammaRegression",
    "GammaSplines",
    "Identity",
    "LeastSquares",
    "LogLinear",
    "Network",
    "PerTime",
    "PositiveMeanNetwork",
    "Powers",
    "QuantileBins",
    "Splines",
    "TimeSplineChebyshev",
    "fitted_mean",
    "flat_cubic",
    "kernel_means",
    "not_a_knot",
    "per_time",
    "per_time_means",
    "spline_basis",
    "spline_penalty",
    "tweedie_loss",
]
