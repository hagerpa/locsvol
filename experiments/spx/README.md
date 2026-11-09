# The SPX study of the paper

This folder reproduces Section 3 of the paper: the leverage factor of a local stochastic
volatility model is calibrated to S&P 500 index options (SPX) on a day with a market event, with
the stochastic volatility parameters fitted four weeks earlier and kept fixed. The study runs for
five pairs of days and two backbones; the paper reports the pair of 20 November and
18 December 2024 with the two-factor Bergomi model. The results for every pair and backbone are
in the folders `<pair>/<backbone>/` and summarized in [Results](#results) below.

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
local volatility follows by the Dupire formula and is read at x clipped to the quantiles of the
SSVI fit at probabilities eps and 1 - eps (below), where prices determine the surface.

## Backbones (Section 3.2)

| | |
|---|---|
| Bergomi | two factors with k_1 = 4, k_2 = 0.12, rho_Z = 0.3; (nu, theta, rho_1, rho_2) fitted on the earlier day by eight regularized Gauss-Newton steps on vega-weighted squared price differences, quotes beyond 18 days; forward variance xi = 1 on the event day; variance factor floored at 10^-10 and divided by its exact mean |
| Heston | (kappa, theta, sigma, rho, v0) fitted on the earlier day by least squares with Fourier prices; quadratic-exponential scheme; variance factor divided by its exact mean |

## Simulation (Section 3.3)

| | |
|---|---|
| Time grid | t_k = (90/365)(k/800)^2 up to 90 days, then steps of 1.1/440 up to 1.1 years, with the quoted maturities inserted (1,159 nodes on 18 Dec 2024) |
| Calibration paths | 32,768 antithetic paths from scrambled Sobol points with a Brownian bridge, for the fixed-point iteration and the particle method; four further sets of 32,768 each for the Bergomi fit and for the correction steps |
| Evaluation | 20 independently scrambled sets of 65,536 such paths for every model |
| Evaluation in the paper's tables | 320 sets of 65,536 independent pseudo-random paths |

## Leverage factor and fixed-point iteration (Sections 2 and 3.3)

| | |
|---|---|
| Target and base | the Dupire local volatility sigma_Dup of the refined SVI surface, read at x clipped to [x_eps(t), x_{1-eps}(t)], the quantiles of X_t at probabilities eps and 1 - eps under the law implied by the SSVI fit, and tabulated at 801 equally spaced values of the state coordinate; eps = kappa / N with N = 32,768 training paths and kappa = 3.2768, so eps = 10^-4; the iteration starts from L0 = sigma_Dup |
| Coordinates | state u = (2/pi) arctan(x / s(t)) with s(t) = theta(max(t, 0.001))^gamma / eta from the SSVI fit; time: s(t) mapped onto [-1, 1] over [0, 1.1] and held below the first grid node; u clipped to [u(t, x_eps(t)), u(t, x_{1-eps}(t))], the images of the same quantiles, and mapped affinely onto [-1, 1]; the quantiles are tabulated at the times of the table and interpolated linearly in t; about three training paths lie beyond each edge at each time, and beyond the edges the leverage factor is constant in x |
| Basis | Chebyshev tensor products of degrees (10, 12), 143 coefficients, and (6, 8), 63 coefficients |
| Regression | Gamma regression with log link, ridge 2.5 * 10^-7 on the coefficients zeta of the regression (the leverage shifts by -zeta/2); every fourth time node of every path, phases by the van der Corput sequence; trapezoidal weights in time |
| Iterations | 12 on the same paths; models after 3, 5, 8 and 12 iterations are evaluated |

## Particle method (Section 3.3)

As published by Guyon and Henry-Labordere: quartic kernel, bandwidth
1.5 sigma_VS(t) sqrt(max(t, 1/4)) N^(-1/5), max(ceil(30 sqrt(t)), 15) nodes between the digital
quantiles of the target at 0.001 and 0.999, not-a-knot cubic spline with flat tails, simulated with
its absolute value; on the same 32,768 paths.

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

## Results

Each folder `<pair>/<backbone>/` holds the results of one run of `run.py`: the backbone (Bergomi or Heston) is fitted to the S&P 500 index options of the earlier day and the leverage factor to those of the event day. Every run is evaluated on 20 sets of 65,536 scrambled Sobol paths.

| Date pair | Backbone | RMSE, (10,12) after 5 iterations | RMSE, (6,8) after 5 iterations | RMSE, particle method | Liquid inside, (10,12) corrected | Liquid inside, (6,8) corrected | Liquid inside, particle method |
|---|---|---|---|---|---|---|---|
| [`svb`](svb/bergomi) (2023-02-13 → 2023-03-13) | [Bergomi](svb/bergomi) | 4.1 / 2.6 / 8.0 | 6.7 / 3.0 / 11.0 | 21.2 / 9.6 / 16.2 | 42% | 45% | 16% |
| [`svb`](svb/heston) (2023-02-13 → 2023-03-13) | [Heston](svb/heston) | 4.5 / 2.4 / 5.0 | 5.6 / 3.0 / 14.4 | 17.9 / 9.5 / 17.1 | 47% | 45% | 18% |
| [`carry-unwind`](carry-unwind/bergomi) (2024-07-08 → 2024-08-05) | [Bergomi](carry-unwind/bergomi) | 6.4 / 4.1 / 5.5 | 6.8 / 3.6 / 4.7 | 31.1 / 23.4 / 12.2 | 98% | 98% | 67% |
| [`carry-unwind`](carry-unwind/heston) (2024-07-08 → 2024-08-05) | [Heston](carry-unwind/heston) | 5.0 / 6.8 / 9.3 | 7.4 / 7.1 / 6.4 | 55.6 / 34.6 / 21.4 | 95% | 95% | 55% |
| [`election`](election/bergomi) (2024-10-09 → 2024-11-06) | [Bergomi](election/bergomi) | 7.7 / 5.7 / 7.1 | 16.3 / 6.0 / 8.0 | 25.9 / 13.4 / 19.8 | 68% | 58% | 30% |
| [`election`](election/heston) (2024-10-09 → 2024-11-06) | [Heston](election/heston) | 9.0 / 8.1 / 13.2 | 21.8 / 11.2 / 17.8 | 38.2 / 20.0 / 19.8 | 73% | 64% | 30% |
| [`fomc`](fomc/bergomi) (2024-11-20 → 2024-12-18) | [Bergomi](fomc/bergomi) | 5.1 / 5.2 / 4.3 | 17.9 / 5.7 / 6.5 | 44.7 / 26.0 / 26.8 | 93% | 87% | 50% |
| [`fomc`](fomc/heston) (2024-11-20 → 2024-12-18) | [Heston](fomc/heston) | 6.5 / 5.5 / 5.7 | 15.1 / 7.5 / 6.8 | 71.3 / 40.0 / 21.4 | 91% | 84% | 40% |
| [`tariffs`](tariffs/bergomi) (2025-02-19 → 2025-04-04) | [Bergomi](tariffs/bergomi) | 3.4 / 3.8 / 4.3 | 4.5 / 3.2 / 5.0 | 31.8 / 17.5 / 32.6 | 99% | 98% | 83% |
| [`tariffs`](tariffs/heston) (2025-02-19 → 2025-04-04) | [Heston](tariffs/heston) | 4.0 / 6.8 / 13.9 | 4.6 / 6.5 / 16.6 | 50.1 / 25.0 / 25.3 | 99% | 99% | 75% |

The RMSE columns give the root mean squared difference between model and target implied volatilities over the left wing, the central region and the right wing, in volatility basis points, for both bases after 5 iterations and for the particle method. The inside columns give the share of liquid quotes inside their bid-ask interval for both bases after four correction steps started after 5 iterations, and for the particle method.

- `carry-unwind`: unwind of the yen carry trade; VIX intraday above 60, SPX about -3%
- `election`: day after the US presidential election; event premium released, SPX about +2.5%
- `fomc`: interest rate decision of the Federal Open Market Committee; SPX about -3%, VIX from 15 to 28
- `svb`: failure of Silicon Valley Bank; front-end rates collapse, VIX 26
- `tariffs`: second day after the tariff announcement of 2 April 2025; SPX about -6%, VIX 45; the earlier day is six weeks before, since March 2025 was itself agitated

Each folder opens with its report, `README.md`: the paper's tables and figures for that date pair and backbone, every model's errors and bid-ask losses, the fixed-point iteration iterate by iterate, every correction chain, and the smiles and prices at every expiry. Besides it:

- `paper/`: the paper's tables (`tables.md`, `tables.json`) and figures (PDF and PNG), from 320 sets of 65,536 independent paths.
- `summary.json`: for every model (the starting point L0, the fixed-point iterates after 3, 5, 8 and 12 iterations of both bases, every correction step and the particle method) the RMSE per region and the bid-ask loss over all and over the liquid quotes, each with its standard error, and the shares of liquid quotes inside, within Monte Carlo noise of and clearly outside their intervals; also the bid-ask statistics of the target surface.
- `parameters.json`: the fitted backbone, the SSVI and refined SVI target, the coordinates and the coefficients of every parametrized leverage factor, with the code commit and the SHA-256 of both data files.
- `figures/`: the figures of the report.

## Running the study

```sh
python experiments/spx/pull_data.py --username <wrds user>          # once, needs WRDS access
python experiments/spx/run.py --pair fomc --backbone bergomi --output <folder>
python experiments/spx/results.py --pair fomc --backbone bergomi --output <folder>
```

`run.py` keeps the results of its expensive steps in the output folder and reuses them when it is
run again. `results.py` writes the results of the run to `<pair>/<backbone>/`: the report
`README.md` with the paper's tables and figures for that pair and backbone and every other model
of the run, `summary.json`, `parameters.json`, and the figures (`report.py`); it also rebuilds
the [Results](#results) section above from every published run.
