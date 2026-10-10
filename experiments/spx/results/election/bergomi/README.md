# election, Bergomi backbone

Day after the US presidential election; event premium released, SPX about +2.5%. The Bergomi backbone is fitted to the S&P 500 index options of 2024-10-09 (nu = 2.3377, theta = 0.2131, rho1 = -0.8464, rho2 = -0.2933); the leverage factor is calibrated to the 2,862 quotes of 2024-11-06, 1,389 of them liquid, at 15 expiries from 9 to 380 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 5.80 ± 3.97 | 5.95 ± 0.85 | 7.50 ± 0.51 |
| Joint regression (6,8), 5 iterations | 13.03 ± 5.58 | 6.06 ± 0.82 | 8.39 ± 0.46 |
| Particle method | 28.91 ± 6.33 | 14.31 ± 1.15 | 19.47 ± 0.46 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 5.03 | 0.95 | 0.54 |
| Joint regression (6,8), 5 iterations | 5.10 | 0.96 | 0.54 |
| Particle method | 4.76 | 0.92 | 0.56 |

### Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 613.33 ± 2.33 | 280.96 ± 0.34 | 1% |
| (10,12) | 3 iterations | 14.62 ± 0.90 | 7.32 ± 0.45 | 28% |
| (10,12) | 5 iterations | 6.45 ± 0.69 | 1.78 ± 0.39 | 70% |
| (10,12) | 8 iterations | 6.39 ± 0.54 | 2.07 ± 0.39 | 65% |
| (6,8) | 0 iterations | 616.33 ± 2.34 | 281.85 ± 0.34 | 1% |
| (6,8) | 3 iterations | 18.22 ± 1.19 | 7.49 ± 0.43 | 29% |
| (6,8) | 5 iterations | 9.88 ± 1.20 | 2.52 ± 0.34 | 61% |
| (6,8) | 8 iterations | 9.55 ± 1.10 | 2.90 ± 0.35 | 55% |
| Particle method |  | 18.76 ± 1.73 | 7.40 ± 0.22 | 31% |
| Refined SVI target |  | 11.00 | 10.98 | 22% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square coefficient increment of the fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 1036.61 ± 3.19 | 369.72 ± 0.31 | 349.05 ± 4.22 | 799.85 ± 2.97 | 327.85 ± 0.41 | 1% | 0% | 99% |
| (10,12) 3 iterations | 47.10 ± 1.97 | 21.87 ± 0.44 | 38.72 ± 1.53 | 27.97 ± 0.80 | 23.49 ± 0.34 | 10% | 2% | 88% |
| (10,12) 5 iterations | 7.69 ± 3.24 | 5.71 ± 0.70 | 7.13 ± 1.38 | 11.56 ± 0.27 | 10.81 ± 0.21 | 25% | 8% | 67% |
| (10,12) 8 iterations | 7.81 ± 3.43 | 1.86 ± 0.51 | 4.20 ± 1.25 | 11.25 ± 0.24 | 10.98 ± 0.18 | 22% | 6% | 72% |
| (10,12) 12 iterations | 7.77 ± 3.42 | 1.95 ± 0.53 | 4.09 ± 1.24 | 11.23 ± 0.25 | 10.95 ± 0.18 | 22% | 7% | 71% |
| (10,12) 0 iterations + 4 correction steps | 858.03 ± 3.05 | 305.44 ± 0.24 | 353.48 ± 3.50 | 612.87 ± 2.48 | 280.77 ± 0.29 | 1% | 0% | 98% |
| (10,12) 3 iterations + 4 correction steps | 19.07 ± 1.87 | 11.35 ± 0.32 | 37.99 ± 1.44 | 14.72 ± 0.65 | 7.53 ± 0.30 | 28% | 6% | 66% |
| (10,12) 5 iterations + 4 correction steps | 10.15 ± 1.32 | 12.51 ± 0.25 | 8.38 ± 1.24 | 6.54 ± 0.54 | 1.89 ± 0.42 | 68% | 11% | 21% |
| (10,12) 8 iterations + 4 correction steps | 8.99 ± 0.92 | 12.30 ± 0.26 | 6.38 ± 1.25 | 6.43 ± 0.46 | 2.15 ± 0.42 | 63% | 15% | 23% |
| (6,8) 3 iterations | 55.04 ± 2.17 | 21.86 ± 0.44 | 47.35 ± 1.63 | 31.02 ± 0.82 | 23.47 ± 0.34 | 11% | 2% | 87% |
| (6,8) 5 iterations | 16.27 ± 2.58 | 6.00 ± 0.70 | 8.03 ± 1.37 | 13.51 ± 0.44 | 11.26 ± 0.21 | 23% | 7% | 70% |
| (6,8) 8 iterations | 14.47 ± 2.70 | 2.50 ± 0.53 | 5.05 ± 1.33 | 13.03 ± 0.43 | 11.46 ± 0.19 | 20% | 7% | 73% |
| (6,8) 12 iterations | 14.55 ± 2.70 | 2.55 ± 0.55 | 4.98 ± 1.33 | 13.03 ± 0.44 | 11.43 ± 0.18 | 20% | 7% | 73% |
| (6,8) 0 iterations + 4 correction steps | 860.73 ± 3.02 | 306.58 ± 0.24 | 354.76 ± 3.46 | 615.83 ± 2.45 | 281.66 ± 0.29 | 1% | 0% | 98% |
| (6,8) 3 iterations + 4 correction steps | 28.59 ± 2.40 | 11.15 ± 0.32 | 45.92 ± 1.67 | 18.60 ± 0.81 | 7.71 ± 0.30 | 28% | 7% | 65% |
| (6,8) 5 iterations + 4 correction steps | 17.43 ± 2.35 | 12.02 ± 0.26 | 9.17 ± 1.22 | 10.38 ± 0.74 | 2.64 ± 0.37 | 58% | 15% | 27% |
| (6,8) 8 iterations + 4 correction steps | 15.81 ± 2.30 | 11.75 ± 0.26 | 6.12 ± 1.03 | 10.03 ± 0.69 | 3.01 ± 0.38 | 52% | 15% | 33% |
| particle method | 25.91 ± 2.09 | 13.44 ± 0.32 | 19.84 ± 1.49 | 17.86 ± 0.45 | 7.35 ± 0.22 | 30% | 8% | 62% |
| refined SVI target |  |  |  | 11.00 | 10.98 | 22% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square coefficient increment of every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 39.33 ± 4.32 | 19.92 ± 0.92 | 34.63 ± 1.36 | 16.74 ± 0.83 | 12.54 ± 0.34 |
| (10,12) 5 iterations | -0.08 ± 0.68 | 3.77 ± 0.23 | 3.04 ± 0.70 | 0.33 ± 0.08 | -0.15 ± 0.18 |
| (10,12) 8 iterations | 0.04 ± 0.05 | -0.08 ± 0.03 | 0.11 ± 0.05 | 0.02 ± 0.00 | 0.03 ± 0.01 |
| (6,8) 3 iterations | 40.49 ± 2.14 | 19.31 ± 0.90 | 42.37 ± 1.91 | 17.99 ± 0.60 | 12.04 ± 0.32 |
| (6,8) 5 iterations | 1.72 ± 0.34 | 3.46 ± 0.29 | 3.05 ± 0.86 | 0.48 ± 0.07 | -0.18 ± 0.17 |
| (6,8) 8 iterations | -0.08 ± 0.03 | -0.05 ± 0.03 | 0.07 ± 0.05 | 0.00 ± 0.00 | 0.03 ± 0.01 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 799.85 ± 2.97 | 327.85 ± 0.41 | 1% |
| 0 iterations | 1 | 724.71 ± 2.67 | 314.77 ± 0.38 | 1% |
| 0 iterations | 2 | 651.40 ± 2.44 | 295.95 ± 0.31 | 1% |
| 0 iterations | 3 | 624.17 ± 2.45 | 286.18 ± 0.29 | 1% |
| 0 iterations | 4 | 612.87 ± 2.48 | 280.77 ± 0.29 | 1% |
| 3 iterations | 0 | 27.97 ± 0.80 | 23.49 ± 0.34 | 10% |
| 3 iterations | 1 | 23.53 ± 0.76 | 18.96 ± 0.32 | 13% |
| 3 iterations | 2 | 18.35 ± 0.69 | 13.16 ± 0.29 | 18% |
| 3 iterations | 3 | 15.72 ± 0.64 | 9.34 ± 0.28 | 25% |
| 3 iterations | 4 | 14.72 ± 0.65 | 7.53 ± 0.30 | 28% |
| 5 iterations | 0 | 11.56 ± 0.27 | 10.81 ± 0.21 | 25% |
| 5 iterations | 1 | 10.16 ± 0.34 | 6.47 ± 0.17 | 29% |
| 5 iterations | 2 | 8.21 ± 0.43 | 3.70 ± 0.32 | 40% |
| 5 iterations | 3 | 7.24 ± 0.51 | 2.58 ± 0.39 | 53% |
| 5 iterations | 4 | 6.54 ± 0.54 | 1.89 ± 0.42 | 68% |
| 8 iterations | 0 | 11.25 ± 0.24 | 10.98 ± 0.18 | 22% |
| 8 iterations | 1 | 10.03 ± 0.28 | 6.84 ± 0.20 | 26% |
| 8 iterations | 2 | 8.14 ± 0.32 | 4.25 ± 0.33 | 39% |
| 8 iterations | 3 | 7.08 ± 0.41 | 2.97 ± 0.40 | 51% |
| 8 iterations | 4 | 6.43 ± 0.46 | 2.15 ± 0.42 | 63% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 799.85 ± 2.97 | 327.85 ± 0.41 | 1% |
| 0 iterations | 1 | 727.45 ± 2.63 | 314.94 ± 0.37 | 1% |
| 0 iterations | 2 | 709.66 ± 2.86 | 301.65 ± 0.34 | 1% |
| 0 iterations | 3 | 629.19 ± 2.41 | 287.41 ± 0.29 | 1% |
| 0 iterations | 4 | 615.83 ± 2.45 | 281.66 ± 0.29 | 1% |
| 3 iterations | 0 | 31.02 ± 0.82 | 23.47 ± 0.34 | 11% |
| 3 iterations | 1 | 26.81 ± 0.80 | 19.03 ± 0.30 | 15% |
| 3 iterations | 2 | 21.64 ± 0.77 | 13.23 ± 0.27 | 18% |
| 3 iterations | 3 | 19.16 ± 0.78 | 9.44 ± 0.28 | 24% |
| 3 iterations | 4 | 18.60 ± 0.81 | 7.71 ± 0.30 | 28% |
| 5 iterations | 0 | 13.51 ± 0.44 | 11.26 ± 0.21 | 23% |
| 5 iterations | 1 | 12.77 ± 0.58 | 6.91 ± 0.18 | 27% |
| 5 iterations | 2 | 11.14 ± 0.67 | 4.18 ± 0.30 | 36% |
| 5 iterations | 3 | 10.70 ± 0.72 | 3.18 ± 0.36 | 47% |
| 5 iterations | 4 | 10.38 ± 0.74 | 2.64 ± 0.37 | 58% |
| 8 iterations | 0 | 13.03 ± 0.43 | 11.46 ± 0.19 | 20% |
| 8 iterations | 1 | 12.34 ± 0.52 | 7.34 ± 0.20 | 25% |
| 8 iterations | 2 | 10.84 ± 0.60 | 4.82 ± 0.31 | 34% |
| 8 iterations | 3 | 10.34 ± 0.66 | 3.65 ± 0.36 | 44% |
| 8 iterations | 4 | 10.03 ± 0.69 | 3.01 ± 0.38 | 52% |

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
| starting point L0 | left wing | 4.1 | 6.6 | 12.5 |
| starting point L0 | central | 1.0 | 1.7 | 2.9 |
| starting point L0 | right wing | 5.9 | 13.4 | 34.4 |
| (10,12) 12 iterations | left wing | 4.1 | 8.2 | 24.0 |
| (10,12) 12 iterations | central | 1.2 | 1.9 | 4.1 |
| (10,12) 12 iterations | right wing | 2.8 | 5.0 | 9.4 |
| particle method | left wing | 3.8 | 7.4 | 19.0 |
| particle method | central | 1.2 | 1.9 | 3.8 |
| particle method | right wing | 2.9 | 5.4 | 8.4 |
