# locSvol

Calibration of local stochastic volatility models by fixed-point iteration, as proposed in

> C. Bayer, D. Belomestny, P. P. Hager and J. Schoenmakers, *Local stochastic volatility
> iterations*, 2026.

A local stochastic volatility model

$$dS_t = L(t, S_t)\sqrt{Y_t}\,S_t\,dW_t$$

combines a stochastic variance factor $Y$, for instance of a Bergomi or a Heston model, with a
leverage factor $L$. It reproduces the implied volatility surface of the market when

$$L(t, s)^2\,\mathbb E[Y_t \mid S_t = s] = \sigma_{\mathrm{Dup}}(t, s)^2,$$

where $\sigma_{\mathrm{Dup}}$ is Dupire's local volatility. locsvol finds $L$ by the fixed-point
iteration $L_{i+1}^2 = \sigma_{\mathrm{Dup}}^2 / \mathbb E_i[Y \mid S]$: each iteration simulates
the model with $L_i$ and approximates the conditional expectation by a single regression in time
and spot. The resulting leverage factor is a smooth function with relatively few coefficients,
which can be refined further against bid and ask quotes.

## Example

[examples/quickstart.py][quickstart] calibrates a two-factor Bergomi model to a
smile made up from a known SSVI surface, so it runs without market data (about a minute on a
laptop). Its core is

```python
import jax.numpy as jnp
from locsvol.backbones import Bergomi
from locsvol.calibration import fixed_point
from locsvol.leverage import Chart, Clipped, Parametrized, quantile_bounds, quantile_envelope, tabulate
from locsvol.regression import ChebyshevTensor, GammaRegression
from locsvol.simulation import simulate
from locsvol.targets import Dupire, fit_ssvi, fit_svi

ssvi = fit_ssvi(maturities, log_strikes, quoted, weights)         # SSVI fit to the quotes
surface = fit_svi(maturities, log_strikes, quoted, weights, ssvi)  # refined slice by slice

times = jnp.linspace(0.0, 1.0, 401)                               # the Euler grid
reference = ssvi.extended(1.0)                                    # theta continued to the horizon
chart = Chart(reference.theta_times, reference.theta_values, reference.eta, reference.gamma, (0.0, 1.0))
bounds = quantile_bounds(reference, times, 2e-4)                  # x_eps(t), x_{1-eps}(t)
local_volatility = tabulate(Clipped(Dupire(surface), *bounds), chart, times)  # sigma_Dup(t, x)
chart = chart.with_envelope(*quantile_envelope(chart, reference, times, 2e-4))

backbone = Bergomi(nu=2.0, theta=0.2, rho1=-0.7, rho2=-0.3)
paths = simulate(backbone, times, dict(seed=1, role="calibration"), 16384)

basis = ChebyshevTensor((6, 8))
start = Parametrized(local_volatility, chart, basis, jnp.zeros(basis.size))  # L = sigma_Dup exp(b^T beta)
iterates = fixed_point(start, local_volatility, paths, GammaRegression(basis, ridge=1e-6), 5)
```

for quotes given as arrays `maturities`, `log_strikes`, `quoted` and `weights` of equal length. The
script uses a time grid graded towards zero and a finer table of the target, and prices the
calibrated model on independent paths: it reproduces the quotes within 12 basis points of implied
volatility.

## Contents

| | |
|---|---|
| Targets | SSVI, eSSVI and SVI surfaces, Dupire local volatility, analytic test surfaces |
| Backbones | two-factor Bergomi, Heston, SABR, constant volatility, and models of your own |
| Regression | Gamma regression with log link, least squares, bins, kernel estimates, neural networks; Chebyshev and spline bases |
| Calibration | fixed-point iteration, particle method of Guyon and Henry-Labordère, correction against bid and ask quotes, fits of the backbone |
| Pricing | Black formula, implied volatility, Monte Carlo prices |

[docs/method.md][method] explains the method step by step, with the functions that
implement each step, and [docs/models.md][models] describes every model. Each module's
folder under `src/locsvol/` has a `README.md` that explains its construction in the terms of the
paper.

## The study of the paper

[experiments/spx][spx] contains the calibration to S&P 500 options of the paper: its
specification, the script that runs it with this package, and the results for every date pair and
backbone. [experiments/essvi][essvi] repeats the study on a parametric target of the
literature, the eSSVI surface of Farkas, Ferrari and Ulrych for SPX on 23 June 2020 with the
reference models of Beiglböck, Hasenbichler and Pammer as backbones.

## Installation

```sh
pip install locsvol
```

The fit of the Heston backbone needs the package roughprix, which the extra `heston` adds
(`pip install "locsvol[heston]"`). To rerun the study of the paper, clone the repository and
install it with the extra `study`, `pip install -e ".[study]"`.

## License

locsvol is licensed under the Apache License 2.0; see [LICENSE][license].

[quickstart]: https://github.com/hagerpa/locsvol/blob/main/examples/quickstart.py
[method]: https://github.com/hagerpa/locsvol/blob/main/docs/method.md
[models]: https://github.com/hagerpa/locsvol/blob/main/docs/models.md
[spx]: https://github.com/hagerpa/locsvol/tree/main/experiments/spx
[essvi]: https://github.com/hagerpa/locsvol/tree/main/experiments/essvi
[license]: https://github.com/hagerpa/locsvol/blob/main/LICENSE
