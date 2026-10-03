# The SPX study of the paper

This folder reproduces Section 3 of the paper: the leverage factor of a local stochastic
volatility model is calibrated to S&P 500 index options (SPX) on a day with a market event, with
the stochastic volatility parameters fitted four weeks earlier and kept fixed. The study runs for
five pairs of days and two backbones; the paper reports the pair of 20 November and
18 December 2024 with the two-factor Bergomi model. Results for every pair and backbone are in
[results](results/README.md).

The study uses only the public functions of the package. `settings.py` holds every number below;
`run.py` runs the study for one date pair and backbone and writes its results.

## Data

End-of-day bid and ask prices of SPX options from OptionMetrics IvyDB US, obtained through
Wharton Research Data Services (WRDS). The data are licensed and not part of the repository.
`date_pairs.py` lists the pairs of days and the SHA-256 of each file, so that anyone with WRDS
access can check that they use the same data; `pull_data.py` downloads them.

| Pair | Earlier day (backbone) | Event day (leverage factor) | Event |
|---|---|---|---|
| `svb` | 13 Feb 2023 | 13 Mar 2023 | failure of Silicon Valley Bank |
| `carry-unwind` | 8 Jul 2024 | 5 Aug 2024 | unwind of the yen carry trade |
| `election` | 9 Oct 2024 | 6 Nov 2024 | day after the US presidential election |
| `fomc` | 20 Nov 2024 | 18 Dec 2024 | interest rate decision of the Federal Open Market Committee |
| `tariffs` | 19 Feb 2025 | 4 Apr 2025 | second day after the tariff announcement of 2 April 2025 |

## Quotes (Section 3.1)

For each maturity T, the forward F(T) and the discount factor D(T) are estimated from put-call
parity by least squares on mid prices. Strikes are represented by k = log(K/F(T)), prices are
divided by D(T)F(T), and out-of-the-money options are used, puts converted to calls. Kept are
quotes with a positive bid, relative spread (ask - bid)/(ask + bid) < 0.1, |k| < 0.2 + 0.4 T and
0.02 < T <= 1.1; per calendar month the expiry with the most quotes, the expiries with T <= 0.05
and at least 150 quotes, and per expiry the more traded of the AM-settled SPX and the PM-settled
SPXW series. Each block of quotes with the same maturity and option type has the same total
weight, shared equally by its quotes. A quote is liquid if the option traded on the day and its
absolute Black delta is at least 0.1.

## Target (Section 3.1)

The power-law SSVI surface of Gatheral and Jacquier is fitted by weighted least squares in implied
volatility under their no-arbitrage conditions, with gamma <= 1/2. Every maturity slice is then
refined by raw SVI under butterfly and calendar constraints. Between maturities the out-of-the-money
prices are interpolated linearly in the square root of the ATM total variance; beyond the last
maturity the total variance grows with slope 0.04; below the first maturity T_1 the surface is
extended by w(t, k) = (t/T_1)(1 + c(1 - t/T_1)) w(T_1, k sqrt((T_1 + t_0)/(t + t_0))), with t_0
and c chosen so that the forward variance of the log-contract is continuous at T_1. The target
local volatility follows by the Dupire formula.

## Backbones (Section 3.2)

| | |
|---|---|
| Bergomi | two factors with k_1 = 4, k_2 = 0.12, rho_Z = 0.3; (nu, theta, rho_1, rho_2) fitted on the earlier day by eight regularized Gauss-Newton steps on vega-weighted squared price differences, quotes beyond 18 days; forward variance xi = 1 on the event day; variance factor floored at 10^-10 and divided by its exact mean |
| Heston | (kappa, theta, sigma, rho, v0) fitted on the earlier day by least squares with Fourier prices; quadratic-exponential scheme; variance factor divided by its exact mean |

## Simulation (Section 3.3)

| | |
|---|---|
| Time grid | t_k = (90/365)(k/800)^2 up to 90 days, then steps of 1.1/440 up to 1.1 years, with the quoted maturities inserted (1,159 nodes on 18 Dec 2024) |
| Calibration paths | 32,768 antithetic paths from scrambled Sobol points with a Brownian bridge |
| Evaluation | 20 independently scrambled sets of 65,536 such paths for every model |
| Evaluation in the paper's tables | 320 sets of 65,536 independent pseudo-random paths |

## Leverage factor and fixed-point iteration (Sections 2 and 3.3)

| | |
|---|---|
| Base L0 | Dupire local volatility capped at 5, read 0.1 inside the domain of the target, tabulated at 801 equally spaced values of the state coordinate |
| Coordinates | state u = (2/pi) arctan(x / s(t)) with s(t) = theta(max(t, 0.001))^gamma / eta from the SSVI fit; time: s(t) mapped onto [-1, 1] over [0, 1.1] and held below the first grid node; u clipped to the running maximum of the 99.9% quantile of abs(u) under L0, capped at 9.2 ATM standard deviations |
| Basis | Chebyshev tensor products of degrees (10, 12), 143 coefficients, and (6, 8), 63 coefficients |
| Regression | Gamma regression with log link, ridge 2.5 * 10^-7 on the coefficients alpha = 2 zeta; every fourth time node of every path, phases by the van der Corput sequence; trapezoidal weights in time |
| Iterations | 12 on the same paths; models after 3, 5, 8 and 12 iterations are evaluated |

## Particle method (Section 3.3)

As published by Guyon and Henry-Labordere: quartic kernel, bandwidth
1.5 sigma_VS(t) sqrt(max(t, 1/4)) N^(-1/5), max(ceil(30 sqrt(t)), 15) nodes between the digital
quantiles of the target at 0.001 and 0.999, not-a-knot cubic spline with flat tails; on the same
32,768 paths.

## Correction against the quotes (Section 3.3)

Four regularized Gauss-Newton steps, started after 0, 3, 5 and 8 iterations, on the vega-scaled
distance of the model prices to the bid-ask intervals; for a quote that is not liquid, the interval
is widened to contain its price at the start. Each step is restricted to a trust region of radius
0.005, 0.01, 0.01, 0.01 in the root mean square change of log L along the paths, 0.03 in total,
with |log L - log L_start| <= 0.1 at 456 x 81 support points; the local problem is solved by SLSQP.
Prices on four sets of 32,768 paths, derivatives by pathwise differentiation.

## Evaluation (Section 3.4)

Implied volatilities of the model at 81 equally spaced log strikes per maturity between the
smallest and largest quoted ones are compared with the target. A point belongs to the left wing if
k <= 0.4 k_min(T), to the right wing if k >= 0.4 k_max(T), to the central region otherwise; within
each region every maturity has the same weight. Reported are the root mean squared differences in
volatility basis points, the bid-ask loss over all and over the liquid quotes, and the share of
liquid quotes inside their bid-ask intervals. Standard errors come from the paired delete-one
jackknife over the sets of paths.

## Running the study

```sh
python experiments/spx/pull_data.py --username <wrds user>          # once, needs WRDS access
python experiments/spx/run.py --pair fomc --backbone bergomi --output <folder>
python experiments/spx/results.py --pair fomc --backbone bergomi --output <folder>
```

`run.py` keeps the results of its expensive steps in the output folder and reuses them when it is
run again. `results.py` writes the results of the run to `results/<pair>/<backbone>/`: the report
`README.md` with the paper's tables and figures for that pair and backbone and every other model
of the run, `summary.json`, `parameters.json`, and the figures (`report.py`).
