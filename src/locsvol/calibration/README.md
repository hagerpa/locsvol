# Calibration

The leverage factor $L$ calibrates the model $\mathrm dS_t = S_t\,L(t,S_t)\sqrt{Y_t}\,\mathrm dW_t$
to the target local volatility $\sigma_\mathrm{Dup}$ when

$$L(t,s)^2\,\mathbb E[Y_t \mid S_t = s] = \sigma_\mathrm{Dup}(t,s)^2,$$

the calibration condition of the paper. This module determines $L$ by the fixed-point iteration,
by the particle method as a reference, and by two variants in between, refines it against the
quotes, and fits the backbones.

## Fixed-point iteration (`fixed_point.py`)

Starting from $L_0$, the iteration maps $L_i$ to
$L_{i+1}(t,s)^2 = \sigma_\mathrm{Dup}(t,s)^2 / \mathbb E_i[Y_t \mid S_t = s]$, where $\mathbb E_i$ is the
expectation in the model with leverage factor $L_i$. For the parametrized leverage factor
$L_\beta = L_0\exp(b^\top\beta)$ of `locsvol.leverage`, one iteration

1. simulates the log spot $X$ on the time grid with $L_i$, on the paths of $Y$ simulated once;
2. fits the Gamma regression of $Y_t$ on the coordinates of $(t, X_t)$ with the offset
   $\log(\sigma_\mathrm{Dup}^2 / L_i^2)$, so that $\mathbb E_i[Y_t \mid X_t = x]$ is approximated by
   $\sigma_\mathrm{Dup}(t,x)^2 / L_i(t,x)^2 \exp(b(t,x)^\top\alpha_i)$;
3. sets $\beta_{i+1} = \beta_i - \alpha_i/2$, that is, $L_{i+1} = L_i\exp(-b^\top\alpha_i/2)$.

The regression is one problem jointly in time and state over all paths. It uses every $k$th
time node of every path, path $m$ starting at the node $\lfloor k v_m\rfloor$ with the van der
Corput sequence $v_m$ in base two, so that every node is used equally often, and each row
carries $k$ times the trapezoidal weight of its time. The study runs twelve iterations with
$k = 4$.

## Particle method (`particle.py`)

The reference method calibrates forward in time on one set of paths, the particles. At every
node $t_j$ it estimates the conditional mean $\mathbb E[Y_{t_j} \mid S_{t_j} = s]$ by the
Nadaraya–Watson regression with the quartic kernel and the bandwidth
$1.5\,\sigma_{\mathrm{VS}}(t_j)\sqrt{\max(t_j, 1/4)}\,N^{-1/5}$ at $\max(\lceil 30\sqrt{t_j}\rceil, 15)$
levels between the digital quantiles of the target at the probabilities $0.001$ and $0.999$,
sets $L(t_j, s) = \sigma_\mathrm{Dup}(t_j, s) / \sqrt{\text{estimate}}$ there, interpolates by the
not-a-knot cubic spline, constant beyond the outer levels, and advances the particles by one
Euler step with the absolute value of that leverage factor, which has the same law. These are
the published settings, with $\sigma_{\mathrm{VS}}$ the variance-swap volatility of the target
(`variance_swap_volatility`). In the terms of the fixed-point iteration, it is the special case
of one regression per time step and as many iterations as time steps.

## Variants (`progressive.py`, `time_interpolated.py`)

The progressive calibration is the particle method with a regression of `locsvol.regression`
in place of the kernel estimate, one fit per node in a single forward pass; with quantile bins
it is the method of van der Stoep, Grzelak and Oosterlee. The time-interpolated iteration keeps
one coefficient vector per knot of the time grid, interpolated linearly in time, and updates the
knots either over the full horizon in every iteration or interval by interval. Both serve for
comparison and are not used in the paper's study.

## Correction against the quotes (`correction.py`, `gauss_newton.py`)

Since $L_\beta$ has relatively few coefficients, it can be calibrated directly to the bid and ask
quotes by Monte Carlo, starting from an iterate of the fixed point. The loss is the vega-scaled
distance of the model prices to the bid–ask intervals; for a liquid quote its interval, for every
other quote the smallest interval containing its bid–ask interval and its price at the start,
so that such a quote may improve but costs only by moving further out (`widened_intervals`).
A correction step linearizes the Monte Carlo prices in $\beta$, with the pathwise derivatives of
`locsvol.pricing`, and minimizes the linearized loss with a penalty in the metric

$$M(\beta) = \frac1m\sum_{k=1}^m\sum_j w_j\,b(t_j, X^k_j)\,b(t_j, X^k_j)^\top,$$

the Gram matrix of the basis along the paths, so that $\delta^\top M\delta$ is the mean-square change
of $\log L$ along them, under a trust region of radius $\rho_i$ in this metric, a cumulative
radius $R$ in the metric of the starting point, and the bound $|\log L_{\beta+\delta} - \log L_{\beta_0}| \le b$
at support points thinned uniformly over the time grid and equally spaced in the state
coordinate (`support_points`). The last bounds are linear in $\beta$, and the local problem is
solved by SLSQP; the step is applied in full. After each step, prices, metric and Jacobian are
recomputed on independent sets of paths. The study applies four steps started after $0$, $3$,
$5$ and $8$ iterations.

## Fits of the backbones (`bergomi_fit.py`, `heston_fit.py`)

The Bergomi parameters $(\nu, \vartheta, \rho_1, \rho_2)$ are fitted to the mid prices of the
earlier day in the pure stochastic volatility model, with the leverage factor $\sqrt{\xi(t)}$ of
that day's forward variance, by regularized Gauss–Newton steps in a trust region on Monte Carlo
prices over fixed Brownian increments; the price derivatives come from forward-mode
differentiation of the simulation, the steps keep the correlation matrix of the three
Brownian motions positive definite, and a step whose solver reaches the minimizer without
certifying its tolerance is taken at a hundred times that tolerance, which stops at the same
point. The Heston parameters are fitted to the same prices by
least squares on Fourier prices of the package `roughprix`.
