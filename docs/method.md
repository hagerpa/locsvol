# The method

This page follows Section 2 of the paper and names, for every step, the code that carries it out.
Paths are relative to `src/locsvol`.

## The model and the calibration condition

A local stochastic volatility (LSV) model is

$$dS_t = S_t\,L(t, S_t)\sqrt{Y_t}\,dW_t,\qquad S_0 > 0,$$

with a stochastic variance factor $Y$, the *backbone*, and the *leverage factor* $L$. With
$X = \log(S/F)$ it is simulated on a time grid $0 = t_0 < \dots < t_n = T$ by the Euler scheme

$$X_{j+1} = X_j - \tfrac12\sigma_j^2 (t_{j+1} - t_j) + \sigma_j (W_{t_{j+1}} - W_{t_j}),\qquad
\sigma_j = L(t_j, X_j)\sqrt{Y_{t_j}}$$

(`simulation/euler.py`). The variance factor does not depend on $L$: its paths are simulated
once and reused for every leverage factor (`simulation/paths.py`).

The model reproduces the implied volatility surface of the market when $S$ has the marginal laws
of Dupire's local volatility model with local volatility $\sigma_{\mathrm{Dup}}$. By Gyöngy's
Markovian projection theorem this holds if

$$L(t, x)\sqrt{\mathbb E[Y_t \mid S_t = x]} = \sigma_{\mathrm{Dup}}(t, x).$$

## The fixed-point iteration

Starting from $L^{(0)}$, the iteration simulates the model with $L^{(i)}$ and sets

$$L^{(i+1)}(t, x) = \frac{\sigma_{\mathrm{Dup}}(t, x)}{\sqrt{\mathbb E[Y_t \mid S^{(i)}_t = x]}}.$$

A fixed point satisfies the calibration condition. The conditional expectation is approximated by
a single regression in time and state over all simulated paths and time nodes,

$$\widehat c^{(i)} \in \operatorname*{argmin}_{g \in \mathcal H} \frac1m \sum_{k=1}^m \sum_{j=0}^n
w_j\, \ell_V\bigl(Y^k_{t_j}, g(t_j, \widehat S^{(i),k}_{t_j})\bigr),$$

with a loss $\ell_V$ whose conditional expectation is minimized by the conditional mean: least
squares for $V \equiv 1$ and the Gamma quasi-likelihood loss $y/\mu + \log\mu$ for $V(u) = u^2$
(`regression/gamma.py`). Algorithm 2.1 is `calibration/fixed_point.py`; each pass of its loop
simulates, regresses and updates.

## The leverage factor and its update

The leverage factor is parametrized as

$$L_\beta(t, x) = L^{(0)}(t, x)\exp\bigl(b(\tau(t), u(t, x))^\top \beta\bigr)$$

with a basis $b$ on $[-1, 1]^2$, for instance the tensor products $T_p(\tau) T_q(u)$ of Chebyshev
polynomials (`regression/chebyshev.py`, `leverage/parametrized.py`). With the regression class
$\mathcal H = \{\sigma_{\mathrm{Dup}}^2 / L_{\beta^{(i)}}^2 \cdot \exp(b^\top \alpha)\}$, a Gamma
regression with log link and the offset $\log(\sigma_{\mathrm{Dup}}^2 / L_{\beta^{(i)}}^2)$, the
update is exact in the parameters:

$$L^{(i+1)} = L_{\beta^{(i)}} \exp(-b^\top \alpha^{(i)} / 2),\qquad
\beta^{(i+1)} = \beta^{(i)} - \alpha^{(i)}/2.$$

Each regression is strictly convex: the loss has second derivative $y e^{-\eta} > 0$ in the
linear predictor $\eta$, and a ridge penalty makes it strictly convex in $\alpha$.

The target $\sigma_{\mathrm{Dup}}$ is read at $x$ clipped to the quantiles $x_\varepsilon(t)$ and
$x_{1-\varepsilon}(t)$ of the coordinates below, so that it is constant in $x$ beyond them, and is
tabulated on a rectangle in $(t, u)$ and interpolated bilinearly (`leverage/table.py`); the
iteration starts from $L^{(0)} = \sigma_{\mathrm{Dup}}$.

## The coordinates

From the SSVI fit of the target, with ATM total variance $\theta(t)$ and parameters $\eta$ and
$\gamma$, let

$$u(t, x) = \frac{2}{\pi}\arctan\frac{x}{s(t)},\qquad s(t) = \frac{\theta(\max(t, t_{\min}))^\gamma}{\eta}.$$

The time coordinate maps $s(t)$ affinely onto $[-1, 1]$ and is held at its value at the first
node of the time grid below it. The state coordinate is $u$ clipped to
$[u_\varepsilon(t), u_{1-\varepsilon}(t)]$ and mapped affinely onto $[-1, 1]$,

$$v(t, x) = \frac{2\,\mathrm{clip}(u, u_\varepsilon, u_{1-\varepsilon}) - u_{1-\varepsilon} - u_\varepsilon}{u_{1-\varepsilon} - u_\varepsilon},\qquad u_\varepsilon(t) = u(t, x_\varepsilon(t)),$$

where $x_\varepsilon(t)$ is the quantile of $X_t$ at probability $\varepsilon$ under the law that the
SSVI fit implies, $\mathbb P(X_t \le x) = \Phi(-d_2) + \varphi(d_2)\,\partial_x\sqrt{w(t, x)}$ with
$d_2 = -x/\sqrt{w} - \sqrt{w}/2$ (`targets/quantiles.py`). The quantiles are tabulated at the
times of the table and interpolated linearly in $t$. With $\varepsilon = \kappa/N$
for $N$ training paths, about $\kappa$ paths lie beyond each edge at each time, and beyond the
edges the leverage factor is constant in $x$ (`leverage/chart.py`).

## Simulation

The Brownian increments come from scrambled Sobol points with a Brownian bridge and antithetic
pairs, or from independent pseudo-random numbers (`simulation/random.py`), on a grid graded near
zero with the quoted maturities as nodes (`simulation/grid.py`). The backbones are the two-factor
Bergomi model, simulated exactly, and the Heston model, by the quadratic-exponential scheme
(`backbones/`). In the LSV model the variance factor is divided by its exact mean, so that
$\mathbb E[Y_t] = 1$ and the leverage factor carries the level of the volatility.

## The particle method

The particle method of Guyon and Henry-Labordère calibrates the leverage factor forward in time:
at each node it estimates $\mathbb E[Y_t \mid S_t = s]$ by kernel regression over the simulated
particles, sets $L = \sigma_{\mathrm{Dup}}/\sqrt{\cdot}$ at a few nodes in $s$, interpolates by a
cubic spline and advances the particles one Euler step (`calibration/particle.py`,
`regression/kernel.py`, `regression/cubic.py`). The paper uses it as a reference.

## Correction against the quotes

A fixed point reproduces the target local volatility, which is itself only a fit to the quotes.
Since $L_\beta$ has relatively few coefficients, it can be refined by a few regularized
Gauss–Newton steps against the bid and ask quotes, each restricted to a trust region in which the
linearized Monte Carlo prices are accurate (`calibration/correction.py`).

## The target

The target local volatility comes from an arbitrage-free fit of the quotes: the power-law SSVI
surface, refined slice by slice by raw SVI, extended below the first maturity, and turned into
$\sigma_{\mathrm{Dup}}$ by the Dupire formula in terms of total variance (`targets/`).
