# Targets

The fixed-point iteration needs a smooth local volatility $\sigma_\mathrm{Dup}$. This module builds it
from option quotes in three steps, a power-law SSVI surface, a refined SVI surface and Dupire's
formula, and provides the parametric surfaces and the functionals of a surface that the
calibration and the evaluation use.

## Quotes (`quotes.py`)

For each expiry, the forward $F$ and the discount factor $D$ follow from put–call parity
$C - P = D(F - K)$ by weighted least squares on the mid prices of the strikes quoted on both
sides. Strikes enter as the log-moneyness $k = \log(K/F)$, prices divided by $DF$, and the
implied volatility of a quote is that of its out-of-the-money option, the put for $k < 0$ and
the call for $k \ge 0$.

## SSVI surface (`ssvi.py`, `ssvi_fit.py`)

The power-law SSVI surface of Gatheral and Jacquier,

$$w(T,k) = \frac{\theta(T)}2\Bigl(1 + \rho\varphi k + \sqrt{(\varphi k + \rho)^2 + 1 - \rho^2}\Bigr), \qquad \varphi = \eta\,\theta(T)^{-\gamma},$$

with the ATM total variance $\theta(T) = w(T,0)$ linear between the expiries, is fitted by weighted
least squares in implied volatility under their sufficient conditions for the absence of
butterfly arbitrage, with $\gamma \le 1/2$; $\theta$ is parametrized by its forward variances so that
it increases. Its derivatives in $k$ and $T$ are closed-form, with the right derivative in $T$ at
the knots. The surface serves as the reference of the chart and of the quantiles
(`locsvol.leverage`) and as the starting point of the refinement.

## Refined SVI surface (`svi.py`, `refined_fit.py`, `refined.py`)

Every expiry slice is refined by the raw SVI parametrization
$w(T,k) = a + b(\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2})$, starting from the SSVI slice, by weighted
least squares in implied volatility subject to the constraints of no butterfly and no calendar
arbitrage on a grid of log-moneyness values, in sweeps over the expiries with the neighbours
held fixed. Between expiries the surface interpolates the out-of-the-money prices of the two
slices linearly in the square root of the ATM total variance and inverts the price in total
variance (`inversion.py`), far in the wings it takes the later slice, beyond the last expiry the
total variance grows with a constant forward variance, and below the first expiry $T_1$ it is
extended by

$$w(t,k) = \frac t{T_1}\Bigl(1 + c\Bigl(1 - \frac t{T_1}\Bigr)\Bigr)\,w\Bigl(T_1,\,k\sqrt{\tfrac{T_1 + t_0}{t + t_0}}\Bigr),$$

with $t_0$ and $c$ chosen so that the forward variance of the log-contract is continuous at
$T_1$ (`short_extension`). The slices meet at the expiries with kinks of the local volatility,
which is why the time grid of the study is fine there.

## Extended SSVI surface (`essvi.py`)

The surface of Hendriks and Martini in the form of Farkas, Ferrari and Ulrych, the SSVI shape
with $\varphi(\theta) = \eta\theta^{-\lambda}(1+\theta)^{\lambda-1}$ and
$\rho(\theta) = \rho_m + (\rho_0 - \rho_m)e^{-a\theta}$, given rather than fitted, with closed-form
derivatives by the chain rule through $\theta$. It is the parametric target of the second study
of the package.

## Dupire's formula (`dupire.py`)

For a surface $w$ free of arbitrage, the local variance in $x = \log(S/F)$ is

$$\sigma_\mathrm{Dup}(t,x)^2 = \frac{\partial_T w}{g}, \qquad g = \Bigl(1 - \frac{x\,\partial_k w}{2w}\Bigr)^2 - \frac{(\partial_k w)^2}4\Bigl(\frac1w + \frac14\Bigr) + \frac{\partial_{kk} w}2,$$

where the density factor $g$ is positive exactly where the smile is free of butterfly
arbitrage. The local variance is undefined wherever $w$, $\partial_T w$, $g$ or their ratio is not
positive and finite, and at $t = 0$ it is replaced by its limit where the surface has one, at the
money for SSVI and at every state for a shifted short extension. Every surface provides its
jet $(w, \partial_T w, \partial_k w, \partial_{kk} w)$ and this limit.

## Functionals of a surface (`log_contract.py`, `quantiles.py`)

The total variance of the log-contract, $T$ times the fair variance-swap rate, is the strip of
out-of-the-money options $\widehat{\mathcal V}(T) = 2\int e^{-k}\,\mathrm{BS}(k, w(T,k))\,\mathrm dk$,
computed by Gauss–Legendre quadrature, and its increments over a time grid give the forward
variance that the Bergomi fit and the particle method use. The distribution function of $X_t$
under a smile is the price of the digital put, $P(X_t \le k) = \Phi(-d_2) + \varphi(d_2)\,\partial_k w/(2\sqrt w)$,
and its quantiles, found by bisection, bound the state coordinate and the reading of the
target in `locsvol.leverage` and place the nodes of the particle method.

## Targets for tests (`analytic.py`, `heston_projection.py`)

A constant volatility, a term structure of forward variance and a synthetic local volatility
have closed-form Dupire local volatilities; the local volatility of a Heston model follows by
Markovian projection, a regression of its variance on the spot over simulated paths. They test
the calibration where the answer is known.
