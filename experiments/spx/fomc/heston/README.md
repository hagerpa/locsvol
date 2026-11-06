# fomc, Heston backbone

Interest rate decision of the Federal Open Market Committee; SPX about -3%, VIX from 15 to 28. The Heston backbone is fitted to the S&P 500 index options of 2024-11-20 (kappa = 4.4825, theta = 0.0391, sigma = 1.4470, rho = -0.6923, v0 = 0.0234); the leverage factor is calibrated to the 2,994 quotes of 2024-12-18, 1,503 of them liquid, at 16 expiries from 9 to 394 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 6.96 ± 4.60 | 6.26 ± 1.00 | 5.81 ± 0.24 |
| Joint regression (6,8), 5 iterations | 13.09 ± 4.45 | 7.68 ± 0.68 | 6.92 ± 0.20 |
| Particle method | 74.21 ± 4.57 | 40.91 ± 1.08 | 21.24 ± 0.22 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 3.53 | 0.90 | 0.52 |
| Joint regression (6,8), 5 iterations | 3.57 | 0.89 | 0.42 |
| Particle method | 3.36 | 0.87 | 0.43 |

### Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 537.74 ± 1.74 | 296.75 ± 0.49 | 2% |
| (10,12) | 3 iterations | 8.55 ± 0.79 | 4.52 ± 0.16 | 49% |
| (10,12) | 5 iterations | 3.37 ± 0.70 | 0.44 ± 0.16 | 91% |
| (10,12) | 8 iterations | 3.45 ± 0.54 | 0.73 ± 0.31 | 80% |
| (6,8) | 0 iterations | 539.03 ± 1.74 | 297.34 ± 0.48 | 2% |
| (6,8) | 3 iterations | 10.67 ± 1.22 | 4.71 ± 0.13 | 51% |
| (6,8) | 5 iterations | 5.59 ± 1.11 | 0.79 ± 0.34 | 86% |
| (6,8) | 8 iterations | 6.28 ± 1.14 | 1.25 ± 0.30 | 76% |
| Refined SVI target |  | 7.37 | 6.40 | 42% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square change of the leverage factor along the simulated paths in every fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 1089.05 ± 5.64 | 405.73 ± 1.02 | 238.21 ± 1.12 | 710.84 ± 3.88 | 374.18 ± 1.09 | 2% | 0% | 98% |
| (10,12) 3 iterations | 53.99 ± 6.00 | 33.57 ± 1.53 | 25.65 ± 1.92 | 29.83 ± 2.44 | 26.52 ± 1.28 | 32% | 3% | 65% |
| (10,12) 5 iterations | 6.54 ± 3.54 | 5.48 ± 1.08 | 5.74 ± 0.54 | 7.78 ± 0.54 | 6.09 ± 0.28 | 49% | 9% | 42% |
| (10,12) 8 iterations | 5.69 ± 4.26 | 4.71 ± 0.49 | 6.21 ± 0.79 | 8.39 ± 0.25 | 7.68 ± 0.30 | 48% | 5% | 47% |
| (10,12) 12 iterations | 5.55 ± 4.09 | 4.68 ± 0.54 | 6.41 ± 0.89 | 8.34 ± 0.30 | 7.66 ± 0.32 | 48% | 5% | 47% |
| (10,12) 0 iterations + 4 correction steps | 861.00 ± 5.08 | 304.19 ± 0.73 | 250.27 ± 1.14 | 535.77 ± 3.08 | 296.23 ± 0.77 | 2% | 0% | 98% |
| (10,12) 3 iterations + 4 correction steps | 17.73 ± 5.01 | 13.80 ± 0.63 | 22.32 ± 1.86 | 9.77 ± 1.61 | 4.81 ± 0.42 | 50% | 11% | 39% |
| (10,12) 5 iterations + 4 correction steps | 12.86 ± 3.27 | 13.27 ± 0.69 | 9.84 ± 0.32 | 4.42 ± 1.43 | 0.47 ± 0.26 | 91% | 8% | 1% |
| (10,12) 8 iterations + 4 correction steps | 10.61 ± 2.85 | 12.95 ± 0.76 | 9.36 ± 0.60 | 4.31 ± 1.24 | 0.74 ± 0.33 | 82% | 14% | 4% |
| (6,8) 3 iterations | 59.25 ± 6.37 | 33.61 ± 1.44 | 23.77 ± 1.27 | 31.26 ± 2.35 | 26.46 ± 1.17 | 28% | 3% | 69% |
| (6,8) 5 iterations | 15.13 ± 5.89 | 7.51 ± 0.60 | 6.76 ± 0.53 | 9.76 ± 1.18 | 6.77 ± 0.25 | 44% | 8% | 48% |
| (6,8) 8 iterations | 13.69 ± 5.23 | 6.83 ± 0.52 | 7.71 ± 0.79 | 10.09 ± 0.93 | 8.19 ± 0.31 | 42% | 5% | 53% |
| (6,8) 12 iterations | 13.66 ± 5.36 | 6.80 ± 0.58 | 7.87 ± 0.83 | 10.06 ± 0.98 | 8.17 ± 0.33 | 42% | 6% | 52% |
| (6,8) 0 iterations + 4 correction steps | 862.56 ± 5.03 | 304.89 ± 0.72 | 250.30 ± 1.15 | 537.07 ± 3.06 | 296.81 ± 0.77 | 2% | 0% | 98% |
| (6,8) 3 iterations + 4 correction steps | 23.99 ± 6.27 | 12.15 ± 0.57 | 20.40 ± 1.27 | 11.99 ± 1.84 | 4.93 ± 0.32 | 52% | 9% | 39% |
| (6,8) 5 iterations + 4 correction steps | 15.77 ± 5.60 | 12.52 ± 0.60 | 10.71 ± 0.67 | 6.78 ± 1.70 | 0.65 ± 0.07 | 84% | 11% | 5% |
| (6,8) 8 iterations + 4 correction steps | 18.08 ± 5.46 | 11.83 ± 0.52 | 10.47 ± 0.79 | 7.44 ± 1.68 | 1.22 ± 0.18 | 76% | 13% | 12% |
| particle method | 71.30 ± 4.89 | 40.00 ± 0.86 | 21.40 ± 0.65 | 35.90 ± 1.85 | 20.63 ± 0.72 | 40% | 5% | 55% |
| refined SVI target |  |  |  | 7.37 | 6.40 | 42% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square change of the leverage factor along the simulated paths in every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 48.44 ± 8.77 | 28.89 ± 1.37 | 19.24 ± 2.74 | 21.49 ± 2.22 | 18.85 ± 1.13 |
| (10,12) 5 iterations | 0.99 ± 3.79 | 0.81 ± 1.25 | -0.67 ± 1.11 | -0.56 ± 0.30 | -1.57 ± 0.40 |
| (10,12) 8 iterations | 0.13 ± 0.48 | 0.03 ± 0.13 | -0.20 ± 0.11 | 0.05 ± 0.08 | 0.02 ± 0.04 |
| (6,8) 3 iterations | 45.59 ± 3.91 | 26.81 ± 1.02 | 15.89 ± 2.07 | 21.20 ± 1.63 | 18.29 ± 1.00 |
| (6,8) 5 iterations | 1.47 ± 1.10 | 0.71 ± 0.86 | -1.12 ± 0.43 | -0.29 ± 0.26 | -1.40 ± 0.34 |
| (6,8) 8 iterations | 0.02 ± 0.22 | 0.03 ± 0.08 | -0.16 ± 0.05 | 0.04 ± 0.06 | 0.02 ± 0.04 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 710.84 ± 3.88 | 374.18 ± 1.09 | 2% |
| 0 iterations | 1 | 634.88 ± 3.57 | 343.79 ± 1.00 | 2% |
| 0 iterations | 2 | 570.13 ± 3.17 | 313.77 ± 0.85 | 2% |
| 0 iterations | 3 | 546.96 ± 3.10 | 302.22 ± 0.79 | 2% |
| 0 iterations | 4 | 535.77 ± 3.08 | 296.23 ± 0.77 | 2% |
| 3 iterations | 0 | 29.83 ± 2.44 | 26.52 ± 1.28 | 32% |
| 3 iterations | 1 | 21.92 ± 2.23 | 18.78 ± 1.13 | 40% |
| 3 iterations | 2 | 13.47 ± 1.76 | 10.54 ± 0.71 | 43% |
| 3 iterations | 3 | 10.59 ± 1.48 | 6.88 ± 0.45 | 47% |
| 3 iterations | 4 | 9.77 ± 1.61 | 4.81 ± 0.42 | 50% |
| 5 iterations | 0 | 7.78 ± 0.54 | 6.09 ± 0.28 | 49% |
| 5 iterations | 1 | 6.82 ± 0.93 | 3.57 ± 0.25 | 54% |
| 5 iterations | 2 | 5.58 ± 1.30 | 1.63 ± 0.13 | 66% |
| 5 iterations | 3 | 4.80 ± 1.44 | 0.75 ± 0.20 | 81% |
| 5 iterations | 4 | 4.42 ± 1.43 | 0.47 ± 0.26 | 91% |
| 8 iterations | 0 | 8.39 ± 0.25 | 7.68 ± 0.30 | 48% |
| 8 iterations | 1 | 7.22 ± 0.43 | 5.15 ± 0.30 | 51% |
| 8 iterations | 2 | 5.62 ± 0.86 | 2.81 ± 0.25 | 58% |
| 8 iterations | 3 | 4.67 ± 1.15 | 1.40 ± 0.25 | 73% |
| 8 iterations | 4 | 4.31 ± 1.24 | 0.74 ± 0.33 | 82% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 710.84 ± 3.88 | 374.18 ± 1.09 | 2% |
| 0 iterations | 1 | 635.92 ± 3.55 | 344.07 ± 1.00 | 2% |
| 0 iterations | 2 | 571.80 ± 3.20 | 314.47 ± 0.86 | 2% |
| 0 iterations | 3 | 548.34 ± 3.08 | 303.03 ± 0.79 | 2% |
| 0 iterations | 4 | 537.07 ± 3.06 | 296.81 ± 0.77 | 2% |
| 3 iterations | 0 | 31.26 ± 2.35 | 26.46 ± 1.17 | 28% |
| 3 iterations | 1 | 22.71 ± 2.13 | 18.29 ± 1.01 | 39% |
| 3 iterations | 2 | 15.19 ± 1.86 | 10.53 ± 0.62 | 42% |
| 3 iterations | 3 | 12.61 ± 1.71 | 6.96 ± 0.38 | 47% |
| 3 iterations | 4 | 11.99 ± 1.84 | 4.93 ± 0.32 | 52% |
| 5 iterations | 0 | 9.76 ± 1.18 | 6.77 ± 0.25 | 44% |
| 5 iterations | 1 | 8.74 ± 1.33 | 3.81 ± 0.29 | 53% |
| 5 iterations | 2 | 7.39 ± 1.51 | 1.95 ± 0.19 | 64% |
| 5 iterations | 3 | 7.77 ± 1.76 | 1.08 ± 0.14 | 77% |
| 5 iterations | 4 | 6.78 ± 1.70 | 0.65 ± 0.07 | 84% |
| 8 iterations | 0 | 10.09 ± 0.93 | 8.19 ± 0.31 | 42% |
| 8 iterations | 1 | 8.97 ± 1.00 | 5.26 ± 0.29 | 49% |
| 8 iterations | 2 | 7.43 ± 1.20 | 3.16 ± 0.27 | 59% |
| 8 iterations | 3 | 6.73 ± 1.39 | 1.86 ± 0.23 | 68% |
| 8 iterations | 4 | 7.44 ± 1.68 | 1.22 ± 0.18 | 76% |

## Smiles at every expiry

![Smiles](figures/smiles.png)

Implied volatility minus the refined SVI target, in bp, with the bid-ask band of the quotes in implied volatility around their mid.

## Prices at every expiry

![Prices](figures/spreads.png)

Model price minus mid price, divided by the vega, in bp, against the bid-ask band; the darker band marks the liquid quotes.

## Monte Carlo uncertainty

Twice the standard error of the pooled implied volatility at the points of the comparison grid, in bp.

| Model | Region | Median | 90th percentile | Maximum |
|---|---|---|---|---|
| starting point L0 | left wing | 6.1 | 9.5 | 17.4 |
| starting point L0 | central | 1.3 | 2.6 | 4.2 |
| starting point L0 | right wing | 2.3 | 4.9 | 7.8 |
| (10,12) 12 iterations | left wing | 7.5 | 12.9 | 20.6 |
| (10,12) 12 iterations | central | 2.0 | 3.2 | 4.4 |
| (10,12) 12 iterations | right wing | 2.2 | 5.0 | 8.6 |
| particle method | left wing | 7.0 | 11.9 | 19.5 |
| particle method | central | 1.9 | 3.0 | 3.8 |
| particle method | right wing | 2.2 | 4.6 | 7.3 |
