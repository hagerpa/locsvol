# election, Heston backbone

Day after the US presidential election; event premium released, SPX about +2.5%. The Heston backbone is fitted to the S&P 500 index options of 2024-10-09 (kappa = 5.4465, theta = 0.0422, sigma = 1.8351, rho = -0.7360, v0 = 0.0300); the leverage factor is calibrated to the 2,862 quotes of 2024-11-06, 1,389 of them liquid, at 15 expiries from 9 to 380 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 18.69 ± 7.29 | 7.17 ± 0.47 | 12.18 ± 0.46 |
| Joint regression (6,8), 5 iterations | 31.96 ± 8.44 | 10.48 ± 0.57 | 16.40 ± 0.50 |
| Particle method | 27.86 ± 6.76 | 18.62 ± 1.10 | 18.59 ± 0.67 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 5.34 | 0.99 | 0.59 |
| Joint regression (6,8), 5 iterations | 5.58 | 1.00 | 0.54 |
| Particle method | 4.89 | 0.94 | 0.58 |

### Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 834.03 ± 2.80 | 308.32 ± 0.39 | 1% |
| (10,12) | 3 iterations | 21.30 ± 1.41 | 17.58 ± 0.59 | 16% |
| (10,12) | 5 iterations | 9.57 ± 1.35 | 1.67 ± 0.18 | 67% |
| (10,12) | 8 iterations | 7.87 ± 0.95 | 1.77 ± 0.14 | 68% |
| (6,8) | 0 iterations | 837.13 ± 2.80 | 308.83 ± 0.39 | 1% |
| (6,8) | 3 iterations | 22.24 ± 1.79 | 17.19 ± 0.56 | 17% |
| (6,8) | 5 iterations | 13.08 ± 2.31 | 2.20 ± 0.17 | 66% |
| (6,8) | 8 iterations | 12.17 ± 2.28 | 2.62 ± 0.13 | 59% |
| Refined SVI target |  | 11.00 | 10.98 | 22% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square change of the leverage factor along the simulated paths in every fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 1224.68 ± 5.34 | 438.79 ± 0.96 | 221.76 ± 3.20 | 1095.19 ± 5.06 | 368.02 ± 1.19 | 1% | 0% | 99% |
| (10,12) 3 iterations | 80.68 ± 8.69 | 54.72 ± 2.10 | 38.75 ± 2.58 | 51.34 ± 3.32 | 50.54 ± 1.82 | 6% | 2% | 92% |
| (10,12) 5 iterations | 9.03 ± 4.11 | 8.14 ± 1.51 | 13.24 ± 1.42 | 12.05 ± 0.33 | 10.44 ± 0.52 | 29% | 16% | 56% |
| (10,12) 8 iterations | 14.25 ± 8.15 | 4.62 ± 1.44 | 15.89 ± 2.06 | 13.02 ± 1.19 | 10.30 ± 0.39 | 28% | 10% | 62% |
| (10,12) 12 iterations | 13.13 ± 7.96 | 3.61 ± 1.15 | 16.46 ± 2.12 | 12.68 ± 0.98 | 10.25 ± 0.43 | 26% | 13% | 62% |
| (10,12) 0 iterations + 4 correction steps | 1028.32 ± 4.64 | 359.55 ± 0.73 | 226.60 ± 3.36 | 836.69 ± 3.89 | 309.31 ± 0.86 | 1% | 0% | 99% |
| (10,12) 3 iterations + 4 correction steps | 23.34 ± 5.16 | 18.42 ± 1.33 | 28.86 ± 2.25 | 18.59 ± 1.85 | 16.29 ± 1.19 | 17% | 6% | 76% |
| (10,12) 5 iterations + 4 correction steps | 6.22 ± 2.58 | 15.16 ± 0.98 | 11.10 ± 1.59 | 7.53 ± 0.50 | 2.04 ± 0.56 | 73% | 17% | 10% |
| (10,12) 8 iterations + 4 correction steps | 11.52 ± 7.20 | 14.86 ± 1.05 | 13.47 ± 2.15 | 7.83 ± 1.22 | 2.37 ± 0.73 | 68% | 17% | 14% |
| (6,8) 3 iterations | 86.82 ± 9.07 | 54.20 ± 2.15 | 37.27 ± 2.46 | 52.28 ± 3.49 | 49.82 ± 1.90 | 5% | 2% | 93% |
| (6,8) 5 iterations | 21.78 ± 5.46 | 11.15 ± 1.30 | 17.85 ± 1.50 | 14.72 ± 1.11 | 10.86 ± 0.44 | 29% | 17% | 55% |
| (6,8) 8 iterations | 22.32 ± 6.29 | 8.70 ± 0.99 | 19.60 ± 1.69 | 15.07 ± 1.29 | 10.60 ± 0.30 | 29% | 12% | 59% |
| (6,8) 12 iterations | 21.74 ± 6.15 | 8.09 ± 0.79 | 20.11 ± 1.71 | 14.77 ± 1.18 | 10.53 ± 0.36 | 27% | 12% | 60% |
| (6,8) 0 iterations + 4 correction steps | 1030.33 ± 4.61 | 360.46 ± 0.72 | 228.40 ± 3.33 | 839.91 ± 3.87 | 309.84 ± 0.85 | 1% | 0% | 99% |
| (6,8) 3 iterations + 4 correction steps | 26.92 ± 6.00 | 18.43 ± 1.24 | 27.32 ± 2.22 | 19.05 ± 1.89 | 15.92 ± 1.16 | 18% | 7% | 75% |
| (6,8) 5 iterations + 4 correction steps | 17.91 ± 5.25 | 15.60 ± 0.99 | 16.12 ± 1.69 | 10.46 ± 1.13 | 2.72 ± 0.58 | 64% | 22% | 14% |
| (6,8) 8 iterations + 4 correction steps | 18.57 ± 5.67 | 15.30 ± 0.98 | 17.30 ± 1.76 | 10.77 ± 1.23 | 3.29 ± 0.68 | 62% | 18% | 20% |
| particle method | 38.03 ± 7.08 | 19.96 ± 1.00 | 19.69 ± 2.66 | 22.35 ± 2.15 | 9.86 ± 0.54 | 29% | 13% | 58% |
| refined SVI target |  |  |  | 11.00 | 10.98 | 22% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square change of the leverage factor along the simulated paths in every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 67.55 ± 16.26 | 51.12 ± 3.12 | 22.29 ± 4.58 | 38.66 ± 4.26 | 40.29 ± 1.64 |
| (10,12) 5 iterations | -4.10 ± 7.31 | 4.53 ± 0.93 | -3.22 ± 0.96 | -0.63 ± 1.03 | 0.19 ± 0.50 |
| (10,12) 8 iterations | 1.12 ± 0.36 | 1.02 ± 0.41 | -0.57 ± 0.09 | 0.34 ± 0.23 | 0.05 ± 0.18 |
| (6,8) 3 iterations | 65.08 ± 12.87 | 46.11 ± 2.79 | 17.16 ± 3.98 | 37.51 ± 4.18 | 39.29 ± 1.70 |
| (6,8) 5 iterations | 0.04 ± 4.18 | 3.06 ± 0.75 | -2.26 ± 0.50 | -0.04 ± 0.83 | 0.33 ± 0.49 |
| (6,8) 8 iterations | 0.59 ± 0.47 | 0.61 ± 0.27 | -0.51 ± 0.05 | 0.30 ± 0.21 | 0.07 ± 0.18 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 1095.19 ± 5.06 | 368.02 ± 1.19 | 1% |
| 0 iterations | 1 | 975.00 ± 4.34 | 349.38 ± 1.08 | 1% |
| 0 iterations | 2 | 877.41 ± 3.98 | 325.43 ± 0.94 | 1% |
| 0 iterations | 3 | 846.28 ± 3.88 | 314.50 ± 0.87 | 1% |
| 0 iterations | 4 | 836.69 ± 3.89 | 309.31 ± 0.86 | 1% |
| 3 iterations | 0 | 51.34 ± 3.32 | 50.54 ± 1.82 | 6% |
| 3 iterations | 1 | 42.95 ± 3.01 | 40.84 ± 1.79 | 8% |
| 3 iterations | 2 | 30.18 ± 2.54 | 28.12 ± 1.62 | 15% |
| 3 iterations | 3 | 22.17 ± 2.04 | 20.04 ± 1.30 | 15% |
| 3 iterations | 4 | 18.59 ± 1.85 | 16.29 ± 1.19 | 17% |
| 5 iterations | 0 | 12.05 ± 0.33 | 10.44 ± 0.52 | 29% |
| 5 iterations | 1 | 11.12 ± 0.35 | 6.67 ± 0.45 | 34% |
| 5 iterations | 2 | 9.19 ± 0.44 | 3.44 ± 0.41 | 44% |
| 5 iterations | 3 | 8.51 ± 0.46 | 2.50 ± 0.62 | 58% |
| 5 iterations | 4 | 7.53 ± 0.50 | 2.04 ± 0.56 | 73% |
| 8 iterations | 0 | 13.02 ± 1.19 | 10.30 ± 0.39 | 28% |
| 8 iterations | 1 | 12.48 ± 1.41 | 6.62 ± 0.44 | 34% |
| 8 iterations | 2 | 10.07 ± 1.31 | 4.10 ± 0.68 | 46% |
| 8 iterations | 3 | 8.86 ± 1.25 | 3.03 ± 0.76 | 58% |
| 8 iterations | 4 | 7.83 ± 1.22 | 2.37 ± 0.73 | 68% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 1095.19 ± 5.06 | 368.02 ± 1.19 | 1% |
| 0 iterations | 1 | 978.95 ± 4.34 | 349.69 ± 1.08 | 1% |
| 0 iterations | 2 | 880.79 ± 4.04 | 325.96 ± 0.94 | 1% |
| 0 iterations | 3 | 849.84 ± 3.92 | 315.06 ± 0.87 | 1% |
| 0 iterations | 4 | 839.91 ± 3.87 | 309.84 ± 0.85 | 1% |
| 3 iterations | 0 | 52.28 ± 3.49 | 49.82 ± 1.90 | 5% |
| 3 iterations | 1 | 43.16 ± 3.20 | 40.14 ± 1.84 | 11% |
| 3 iterations | 2 | 30.57 ± 2.70 | 27.54 ± 1.63 | 18% |
| 3 iterations | 3 | 23.05 ± 2.12 | 19.71 ± 1.30 | 17% |
| 3 iterations | 4 | 19.05 ± 1.89 | 15.92 ± 1.16 | 18% |
| 5 iterations | 0 | 14.72 ± 1.11 | 10.86 ± 0.44 | 29% |
| 5 iterations | 1 | 13.07 ± 0.96 | 6.83 ± 0.41 | 35% |
| 5 iterations | 2 | 12.44 ± 1.32 | 3.85 ± 0.37 | 42% |
| 5 iterations | 3 | 11.06 ± 1.12 | 3.15 ± 0.58 | 56% |
| 5 iterations | 4 | 10.46 ± 1.13 | 2.72 ± 0.58 | 64% |
| 8 iterations | 0 | 15.07 ± 1.29 | 10.60 ± 0.30 | 29% |
| 8 iterations | 1 | 13.93 ± 1.21 | 6.92 ± 0.38 | 37% |
| 8 iterations | 2 | 12.85 ± 1.40 | 4.73 ± 0.60 | 45% |
| 8 iterations | 3 | 11.14 ± 1.18 | 3.89 ± 0.70 | 55% |
| 8 iterations | 4 | 10.77 ± 1.23 | 3.29 ± 0.68 | 62% |

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
| starting point L0 | left wing | 6.2 | 9.4 | 12.0 |
| starting point L0 | central | 1.6 | 3.2 | 4.5 |
| starting point L0 | right wing | 4.6 | 8.5 | 12.1 |
| (10,12) 12 iterations | left wing | 10.9 | 18.6 | 28.3 |
| (10,12) 12 iterations | central | 2.3 | 4.6 | 7.2 |
| (10,12) 12 iterations | right wing | 3.2 | 6.1 | 8.9 |
| particle method | left wing | 10.3 | 17.3 | 24.3 |
| particle method | central | 2.2 | 4.4 | 6.6 |
| particle method | right wing | 3.3 | 6.2 | 9.3 |
