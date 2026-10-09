# Backbones

A backbone is the stochastic variance of the local stochastic volatility model

$$\mathrm dS_t = S_t\,L(t,S_t)\sqrt{Y_t}\,\mathrm dW_t,$$

given to the model as the variance factor $Y$, which has unit mean, $\mathbb E[Y_t] = 1$, so that
the leverage factor $L$ carries the level and the term structure of the volatility and the
backbone only its dynamics. A backbone is any object with the number of Brownian motions it
uses, the spot's first among them, and a method that turns their increments on a time grid into
the paths of $Y$ at the nodes (`custom.py`). The paths of $Y$ do not depend on $L$, so they are
simulated once and reused for every leverage factor.

## Two-factor Bergomi model (`bergomi.py`)

The variance factor of the paper's study,

$$Y_t = \xi(t)\exp\Bigl(\tfrac{2\nu}{\sqrt q}\bigl((1-\vartheta)X^1_t + \vartheta X^2_t\bigr) - \tfrac{v(t)}2\Bigr),$$

with the Ornstein–Uhlenbeck factors $\mathrm dX^a_t = -\kappa_a X^a_t\,\mathrm dt + \mathrm dZ^a_t$,
the correlations $\mathrm d\langle W, Z^a\rangle_t = \rho_a\,\mathrm dt$ and
$\mathrm d\langle Z^1, Z^2\rangle_t = \rho_Z\,\mathrm dt$, the normalization
$q = (1-\vartheta)^2 + \vartheta^2 + 2\rho_Z\vartheta(1-\vartheta)$ and the variance $v(t)$ of the
exponent, so that $\mathbb E[Y_t] = \xi(t)$ for the forward variance curve $\xi$, piecewise
constant (`ForwardVariance`) and equal to one in the calibrated model. The factors are simulated
exactly: over a step of length $h$ the vector of the spot's increment and the two integrated
factor increments is Gaussian with the covariance of the module docstring, and it is generated
from three independent increments by the Cholesky factor of that covariance, whose first row
leaves the spot's increment unchanged. The factor is floored at a small multiple of $\xi(t)$
and divided by the exact mean of the floored lognormal variable, which keeps $\mathbb E[Y_t] = 1$
exactly.

## Heston model (`heston.py`)

The CIR variance $\mathrm dV_t = \kappa(\theta - V_t)\,\mathrm dt + \sigma\sqrt{V_t}\,\mathrm dZ_t$
with $\mathrm d\langle W, Z\rangle_t = \rho\,\mathrm dt$, simulated by the quadratic-exponential
scheme of Andersen, which matches the exact conditional mean and variance of $V_{t+h}$ given
$V_t$ with a squared shifted normal or an exponential law with an atom at zero, depending on
their ratio. The variance factor is $Y_t = V_t / \mathbb E[\max(V_t, \text{floor})]$ with the exact
mean of the floored noncentral chi-square law of $V_t$.

## Further backbones

The constant backbone (`constant.py`) has $Y \equiv 1$, so the calibrated model is the local
volatility model and the calibrated leverage factor is $\sigma_\mathrm{Dup}$ itself; it serves
as a test. The SABR backbone with $\beta = 1$ (`sabr.py`) has a lognormal volatility simulated
exactly. A backbone of the caller (`custom.py`) is built from a Markov scheme for its state,
a function of the state giving the variance, and the mean of that variance, from which the
package forms $Y$; its parameters are arrays, so that compiled code can differentiate the
paths with respect to them, as the fit of the Bergomi model does (`locsvol.calibration`).
