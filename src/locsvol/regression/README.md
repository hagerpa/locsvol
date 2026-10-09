# Regression

The fixed-point iteration approximates the conditional mean of the variance factor given the
spot, $\mathbb E[Y_t \mid X_t = x]$, by one regression jointly in time and state over all simulated
paths. This module provides that regression, the basis it uses, and the alternatives the
particle method and the variants need.

## Gamma regression with log link (`gamma.py`, `gamma_start.py`, `tweedie.py`)

The conditional mean of a positive response $y$ given coordinates $x$ is modelled as
$\mu(x) = \exp(o + b(x)^\top\alpha)$ with a known offset $o$, in the fixed point the logarithm of
$\sigma_\mathrm{Dup}^2/L^2$, and a basis $b$. The coefficients minimize the Gamma quasi-likelihood
loss with a ridge penalty,

$$F(\alpha) = \frac{\sum_n w_n\bigl(\eta_n + y_n e^{-\eta_n}\bigr)}{\sum_n w_n} + \frac\lambda2|\alpha|^2, \qquad \eta_n = o_n + b(x_n)^\top\alpha.$$

Since $y e^{-\eta} > 0$ is the second derivative of the summand in $\eta$, the problem is strictly
convex for $\lambda > 0$ and has a unique minimizer, which keeps the approximated mean positive.
It is found from the ridge regression of $\log y - o$ on $b(x)$, whose matrix $H = G + \lambda I$ with
the Gram matrix $G$ is close to the Hessian near the minimizer, by L-BFGS-B first in the
coordinates $H^{1/2}\alpha$, in which $F$ is nearly isotropic, then in $\alpha$, and finally by damped
Newton steps $H^{-1}\nabla F$ until the largest component of the gradient is below the
tolerance, $10^{-8}$ in the study. The sums over observations run over blocks of rows, so that
memory stays bounded for tens of millions of rows. The Gamma loss is the Tweedie loss of
power two; other powers between one and two are available.

## Bases (`chebyshev.py`, `polynomials.py`, `splines.py`, `mixed.py`, `custom.py`)

The basis of the study is the tensor product of Chebyshev polynomials on $[-1,1]^2$ of
time–state degrees $(p, q)$,

$$b_{i(q+1)+j}(\tau, u) = T_i(\tau)\,T_j(u), \qquad 0 \le i \le p,\ 0 \le j \le q,$$

$63$ functions for $(6,8)$ and $143$ for $(10,12)$. Since $2T_aT_b = T_{a+b} + T_{|a-b|}$, the Gram
matrix of the basis follows from $(2p+1)(2q+1)$ moments instead of a sum of outer products,
which makes the start of the regression and the metric of the correction steps cheap. Further
bases are Chebyshev polynomials of the state alone, monomials, B-splines in one or both
coordinates with a difference penalty on their coefficients, B-splines in time with Chebyshev
polynomials in the state, and any function of the caller.

## Other estimators of the conditional mean

The Nadaraya–Watson kernel regression with the quartic kernel (`kernel.py`) is the estimator
of the particle method; quantile bins with a monotone cubic through the bin means, or the bins
of van der Stoep, Grzelak and Oosterlee (`bins.py`), weighted least squares by the truncated
singular value decomposition (`least_squares.py`), and a neural network with a positive output
(`network.py`) serve the progressive calibration and comparisons. `means.py` collects fitted
means that are piecewise constant in time, one fit per node, as the progressive calibration
produces them.
