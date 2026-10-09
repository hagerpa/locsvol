# Pricing

Prices in the package are divided by the discount factor and the forward of their maturity, so
that a call or put with log strike $k = \log(K/F)$ is a function of $k$ and of the total variance
$w = \sigma^2T$ alone.

## Black's formula (`black.py`)

For a call ($s = 1$) or a put ($s = -1$),

$$p(w,k) = s\bigl(\Phi(s d_1) - e^k\Phi(s d_2)\bigr), \qquad d_1 = -\frac{k}{\sqrt w} + \frac{\sqrt w}2, \quad d_2 = d_1 - \sqrt w,$$

and the vega, the derivative of $p$ in $\sigma$, is $\sqrt T\,\varphi(d_1)$ for calls and puts alike;
divided by the vega, a price difference is to first order a difference in implied volatility,
which is how the study scales every loss on prices. The implied volatility of a price inverts
$p$ in $\sigma$ by bisection, since $p$ increases from the intrinsic value; a price outside the
range of the formula has none.

## Monte Carlo prices (`monte_carlo.py`)

For paths $X^1, \ldots, X^m$ of $X = \log(S/F)$ from the Euler scheme on a grid that contains the
maturity $T$,

$$P \approx \frac1m\sum_{n=1}^m\bigl(s(e^{X^n_T} - e^k)\bigr)^+,$$

for all quotes and comparison strikes of a maturity at once. The paths of one scrambled Sobol
set are not independent, so independent scrambles are averaged as replications
(`replicated_prices`), and the paper's tables use independent pseudo-random paths
(`locsvol.simulation`).

The correction steps need the derivatives of the prices with respect to the coefficients
$\beta$ of the leverage factor. They are pathwise: with $D_j = \partial X_j/\partial\beta$ along a path,

$$\frac{\partial P}{\partial\beta} \approx \frac1m\sum_n s\,e^{X^n_T}\,\mathbf 1\{s(e^{X^n_T} - e^k) > 0\}\,D^n_T,$$

and $D$ follows by differentiating the Euler scheme step by step,

$$D_{j+1} = D_j + (\Delta W_j - \sigma_j h_j)\sqrt{Y_j}\,\bigl(\partial_x L(t_j,X_j)\,D_j + \partial_\beta L(t_j,X_j)\bigr), \qquad D_0 = 0,$$

with $\sigma_j = L(t_j, X_j)\sqrt{Y_j}$ and $h_j = t_{j+1} - t_j$. For $L_\beta = L_0\exp(b^\top\beta)$ the
derivative in $\beta$ is $L_\beta\,b$, and $\partial_x L$ is taken once per path and step, so that each
further coefficient adds only a few arithmetic operations per path and step. The sums over
paths run over chunks of paths, so that memory stays bounded.
