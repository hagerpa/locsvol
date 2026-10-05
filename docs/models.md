# Models

Every model of the package, by subpackage. The paper uses the SVI target, the Bergomi and Heston
backbones, the chart, the Chebyshev tensor basis with the Gamma regression, the fixed-point
iteration, the correction steps and the particle method; the other models are alternatives for
the same steps. Each module docstring gives the full definition.

Throughout, $x = \log(S/F)$ is the log-moneyness of the spot, $Y$ the variance factor with
$\mathbb E[Y_t] = 1$, and the model is $dS_t = L(t, S_t)\sqrt{Y_t}\,S_t\,dW_t$.

## Targets (`locsvol.targets`)

The target is a total variance surface $w(t, k)$ and its Dupire local volatility.

| Model | Definition |
|---|---|
| `SSVI` | Power-law SSVI of Gatheral and Jacquier, $w = \tfrac{\theta}{2}(1 + \rho\varphi k + \sqrt{(\varphi k + \rho)^2 + 1 - \rho^2})$, $\varphi = \eta\theta^{-\gamma}$, with $\theta(t)$ linear between knots; fitted by `fit_ssvi` |
| `RefinedSVI` | A raw SVI slice at every expiry, interpolated in price between expiries, with a short extension below the first expiry (`short_extension`) and constant forward variance beyond the last; fitted by `fit_svi` from an SSVI fit |
| `Dupire` | $\sigma_{\mathrm{Dup}}(t, x)$ of any surface, with the derivatives of $w$ by automatic differentiation |
| `ConstantVolatility`, `TermStructure` | $\sigma_{\mathrm{Dup}} = \sigma$, and $\sigma_{\mathrm{Dup}}(t, x) = \sqrt{\xi(t)}$ for a piecewise constant forward variance |
| `Synthetic` | A closed-form local volatility for tests |
| `HestonProjection` | The local volatility $\sqrt{\mathbb E[V_t \mid X_t = x]}$ of a Heston model, by regression (`heston_projection`) |

Helpers on quotes and surfaces: `put_call_parity` (discount factor and forward per expiry),
`log_moneyness`, `otm_implied_volatility`, `log_contract_variance` and `forward_variance`,
`digital_probability` and `digital_quantiles`.

## Backbones (`locsvol.backbones`)

A backbone simulates the variance factor $Y$ at the nodes of a time grid, normalized to mean one,
so that the leverage factor carries the level of the volatility.

| Model | Variance factor |
|---|---|
| `Bergomi` | Two-factor Bergomi, $Y_t = \xi(t)\exp\big(\tfrac{2\nu}{\sqrt q}((1-\theta)X^1_t + \theta X^2_t) - v(t)/2\big)$ with Ornstein–Uhlenbeck factors $X^a$ of mean reversion $k_a$ and correlations $\rho_1, \rho_2, \rho_Z$, simulated exactly; `ForwardVariance` gives a piecewise constant $\xi$ |
| `Heston` | CIR variance $dV = \kappa(\theta - V)dt + \sigma\sqrt V dZ$, $d\langle W, Z\rangle = \rho\,dt$, by Andersen's quadratic-exponential scheme |
| `SABR` | Lognormal volatility $\alpha_t = \alpha_0 e^{l_t}$, $dl = \nu dZ - \nu^2 dt/2$ ($\beta = 1$), simulated exactly |
| `Constant` | $Y = 1$: the local volatility model |
| `CustomBackbone` | Any Markov scheme $s_{j+1} = \mathrm{transition}(s_j, t_j, h_j, \Delta W_j)$ of the caller with $V_j = \mathrm{state\_variance}(s_j, t_j)$ |

The fits `fit_bergomi` (Gauss–Newton trust region), `fit_bergomi_interior_point` and `fit_heston`
(`locsvol.calibration`) calibrate a backbone to the quotes.

## Simulation (`locsvol.simulation`)

`graded_times` and `with_dates` build the time grid, refined near $t = 0$ and containing the
maturities. `simulate` draws the variance factor and the spot's Brownian increments by randomized
Sobol points with a Brownian bridge and antithetic paths, seeded by a dictionary that names the
stream, and returns `Paths`. `log_spot` runs the Euler scheme of $X$ for a leverage factor
(`euler_step` is one step).

## Leverage factors (`locsvol.leverage`)

| Model | Definition |
|---|---|
| `Chart` | The coordinates $(\tau, u)$ of the paper: $u = \tfrac{2}{\pi}\arctan(x / s(t))$ with the SSVI scale $s(t) = \theta(t)^\gamma/\eta$, and $\tau$ affine in $s(t)$; with an envelope, $u$ clipped to it and mapped onto $[-1, 1]$; `quantile_envelope` places the envelope at the quantiles of a surface |
| `Affine`, `Bounded`, `CustomChart` | Other coordinates: affine in $(t, x)$, arctan with a fixed scale, or a function of the caller |
| `Table`, `tabulate` | A positive function tabulated on the chart's state coordinate at the times of `table_times`, interpolated in between |
| `Capped` | $\min(\sigma(t, x), c)$ read inside log-moneyness bounds; the base $L_0 = \min(\sigma_{\mathrm{Dup}}, 5)$ |
| `Parametrized` | $L_\beta = L_0 \exp(b(\tau, u)^\top \beta)$ |

## Regression (`locsvol.regression`)

A regression approximates $\mathbb E[Y_t \mid X_t]$ from the simulated paths.

| Model | Definition |
|---|---|
| `GammaRegression` | Gamma quasi-likelihood with log link, offset and ridge penalty on a basis; strictly convex, solved to a gradient tolerance; a power $p \ne 2$ gives the Tweedie loss |
| `LeastSquares` | Weighted least squares on a basis |
| `GammaSplines`, `Splines`, `TimeSplineChebyshev` | Penalized B-splines placed by the observations, tensor B-splines, and B-splines in time times Chebyshev polynomials in the state |
| `QuantileBins`, `Bins` | Means in quantile bins of the spot, interpolated (van der Stoep, Grzelak and Oosterlee) |
| `kernel_means` | Nadaraya–Watson estimate with a quartic or Gaussian kernel |
| `PositiveMeanNetwork`, `Network` | A neural network with a positive output |
| `CustomRegression` | A regression of the caller |

Bases: `ChebyshevTensor` (degrees $(p, q)$ in $(\tau, u)$, the basis of the paper), `Chebyshev`,
`Powers`, `Identity`, `CustomBasis`. `per_time_means` fits one regression per time node.

## Calibration (`locsvol.calibration`)

| Method | Definition |
|---|---|
| `fixed_point` | The fixed-point iteration of the paper: simulate with $L_\beta$, fit the Gamma regression of $Y$ with offset $\log(\sigma_{\mathrm{Dup}}^2 / L_\beta^2)$, and set $\beta \leftarrow \beta - \alpha/2$ |
| `correction_steps` | Gauss–Newton steps of $\beta$ against the bid-ask intervals, in a trust region of the mean-square change of $\log L$, with bounds at `support_points` |
| `particle_method` | The particle method of Guyon and Henry-Labordère with the settings of their book: quartic kernel, not-a-knot spline, nodes between digital quantiles; `ParticleLeverage` is its leverage factor |
| `progressive` | One regression per time step in a single forward pass (`Progressive`) |
| `full_horizon`, `interval_progressive` | The fixed-point iteration with coefficients at time knots, interpolated linearly in between (`TimeInterpolated`), on the full horizon or interval by interval |

## Pricing (`locsvol.pricing`)

`black_price`, `black_vega` and `implied_volatility` (bisection) on normalized prices;
`monte_carlo_prices` prices European options under the model with a leverage factor, differentiable
in its coefficients, and `replicated_prices` averages over independent sets of paths.

## Saving

`locsvol.save(path, obj)` and `locsvol.load(path)` write and read every model above, and
dictionaries of them.
