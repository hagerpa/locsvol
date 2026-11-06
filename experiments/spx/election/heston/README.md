# election, Heston backbone

Day after the US presidential election; event premium released, SPX about +2.5%. The Heston backbone is fitted to the S&P 500 index options of 2024-10-09 (kappa = 5.4463, theta = 0.0422, sigma = 1.8350, rho = -0.7360, v0 = 0.0300); the leverage factor is calibrated to the 2,862 quotes of 2024-11-06, 1,389 of them liquid, at 15 expiries from 9 to 380 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 18.69 ± 7.29 | 7.17 ± 0.47 | 12.18 ± 0.46 |
| Joint regression (6,8), 5 iterations | 31.96 ± 8.44 | 10.48 ± 0.57 | 16.38 ± 0.50 |
| Particle method | 27.87 ± 6.76 | 18.62 ± 1.10 | 18.18 ± 0.67 |

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
| (10,12) | 0 iterations | 835.74 ± 2.80 | 308.47 ± 0.39 | 1% |
| (10,12) | 3 iterations | 21.31 ± 1.42 | 17.58 ± 0.59 | 16% |
| (10,12) | 5 iterations | 9.57 ± 1.35 | 1.67 ± 0.18 | 67% |
| (10,12) | 8 iterations | 7.86 ± 0.95 | 1.77 ± 0.14 | 68% |
| (6,8) | 0 iterations | 837.04 ± 2.80 | 308.84 ± 0.39 | 1% |
| (6,8) | 3 iterations | 22.24 ± 1.79 | 17.19 ± 0.56 | 17% |
| (6,8) | 5 iterations | 13.08 ± 2.31 | 2.20 ± 0.17 | 66% |
| (6,8) | 8 iterations | 12.17 ± 2.29 | 2.62 ± 0.13 | 59% |
| Refined SVI target |  | 11.00 | 10.98 | 22% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square change of the leverage factor along the simulated paths in every fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 1224.67 ± 5.35 | 438.79 ± 0.96 | 221.77 ± 3.20 | 1095.18 ± 5.06 | 368.02 ± 1.19 | 1% | 0% | 99% |
| (10,12) 3 iterations | 80.69 ± 8.70 | 54.72 ± 2.10 | 38.79 ± 2.58 | 51.34 ± 3.33 | 50.53 ± 1.82 | 6% | 2% | 92% |
| (10,12) 5 iterations | 9.03 ± 4.11 | 8.14 ± 1.51 | 13.23 ± 1.42 | 12.05 ± 0.33 | 10.44 ± 0.52 | 29% | 16% | 56% |
| (10,12) 8 iterations | 14.25 ± 8.15 | 4.62 ± 1.44 | 15.86 ± 2.05 | 13.01 ± 1.19 | 10.30 ± 0.38 | 28% | 10% | 62% |
| (10,12) 12 iterations | 13.13 ± 7.97 | 3.61 ± 1.15 | 16.43 ± 2.11 | 12.68 ± 0.98 | 10.25 ± 0.43 | 26% | 13% | 62% |
| (10,12) 0 iterations + 4 correction steps | 1029.40 ± 4.64 | 360.06 ± 0.73 | 226.48 ± 3.35 | 838.40 ± 3.89 | 309.46 ± 0.86 | 1% | 0% | 99% |
| (10,12) 3 iterations + 4 correction steps | 23.35 ± 5.17 | 18.42 ± 1.33 | 28.91 ± 2.25 | 18.60 ± 1.85 | 16.29 ± 1.19 | 17% | 7% | 76% |
| (10,12) 5 iterations + 4 correction steps | 6.22 ± 2.58 | 15.15 ± 0.98 | 11.09 ± 1.58 | 7.53 ± 0.50 | 2.03 ± 0.56 | 73% | 17% | 10% |
| (10,12) 8 iterations + 4 correction steps | 11.50 ± 7.20 | 14.86 ± 1.05 | 13.45 ± 2.15 | 7.82 ± 1.21 | 2.36 ± 0.73 | 68% | 17% | 14% |
| (6,8) 3 iterations | 86.83 ± 9.08 | 54.20 ± 2.15 | 37.31 ± 2.46 | 52.28 ± 3.50 | 49.82 ± 1.90 | 5% | 2% | 93% |
| (6,8) 5 iterations | 21.78 ± 5.46 | 11.15 ± 1.30 | 17.82 ± 1.50 | 14.72 ± 1.11 | 10.86 ± 0.44 | 29% | 17% | 55% |
| (6,8) 8 iterations | 22.32 ± 6.29 | 8.70 ± 0.99 | 19.55 ± 1.68 | 15.06 ± 1.29 | 10.60 ± 0.30 | 29% | 12% | 59% |
| (6,8) 12 iterations | 21.73 ± 6.15 | 8.09 ± 0.79 | 20.07 ± 1.71 | 14.76 ± 1.19 | 10.53 ± 0.36 | 27% | 12% | 60% |
| (6,8) 0 iterations + 4 correction steps | 1030.30 ± 4.61 | 360.44 ± 0.72 | 228.40 ± 3.33 | 839.82 ± 3.88 | 309.84 ± 0.85 | 1% | 0% | 99% |
| (6,8) 3 iterations + 4 correction steps | 26.93 ± 6.01 | 18.43 ± 1.24 | 27.36 ± 2.22 | 19.06 ± 1.89 | 15.92 ± 1.16 | 18% | 7% | 75% |
| (6,8) 5 iterations + 4 correction steps | 17.93 ± 5.25 | 15.60 ± 0.99 | 16.11 ± 1.69 | 10.46 ± 1.14 | 2.71 ± 0.58 | 64% | 22% | 14% |
| (6,8) 8 iterations + 4 correction steps | 18.58 ± 5.67 | 15.29 ± 0.98 | 17.26 ± 1.75 | 10.77 ± 1.23 | 3.29 ± 0.68 | 62% | 18% | 20% |
| particle method | 38.16 ± 7.13 | 19.98 ± 1.02 | 19.81 ± 2.73 | 22.39 ± 2.21 | 9.85 ± 0.56 | 30% | 13% | 57% |
| refined SVI target |  |  |  | 11.00 | 10.98 | 22% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square change of the leverage factor along the simulated paths in every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 67.56 ± 16.28 | 51.11 ± 3.13 | 22.36 ± 4.58 | 38.66 ± 4.26 | 40.28 ± 1.64 |
| (10,12) 5 iterations | -4.09 ± 7.32 | 4.53 ± 0.93 | -3.20 ± 0.97 | -0.63 ± 1.03 | 0.19 ± 0.50 |
| (10,12) 8 iterations | 1.12 ± 0.36 | 1.02 ± 0.41 | -0.57 ± 0.09 | 0.34 ± 0.23 | 0.05 ± 0.18 |
| (6,8) 3 iterations | 65.10 ± 12.88 | 46.11 ± 2.80 | 17.24 ± 3.98 | 37.52 ± 4.18 | 39.29 ± 1.70 |
| (6,8) 5 iterations | 0.04 ± 4.19 | 3.06 ± 0.75 | -2.25 ± 0.50 | -0.04 ± 0.83 | 0.33 ± 0.49 |
| (6,8) 8 iterations | 0.59 ± 0.47 | 0.61 ± 0.27 | -0.51 ± 0.05 | 0.30 ± 0.21 | 0.07 ± 0.18 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 1095.18 ± 5.06 | 368.02 ± 1.19 | 1% |
| 0 iterations | 1 | 975.01 ± 4.34 | 349.38 ± 1.08 | 1% |
| 0 iterations | 2 | 877.36 ± 3.98 | 325.43 ± 0.94 | 1% |
| 0 iterations | 3 | 846.26 ± 3.88 | 314.50 ± 0.87 | 1% |
| 0 iterations | 4 | 838.40 ± 3.89 | 309.46 ± 0.86 | 1% |
| 3 iterations | 0 | 51.34 ± 3.33 | 50.53 ± 1.82 | 6% |
| 3 iterations | 1 | 42.95 ± 3.01 | 40.84 ± 1.79 | 8% |
| 3 iterations | 2 | 30.18 ± 2.54 | 28.12 ± 1.62 | 15% |
| 3 iterations | 3 | 22.18 ± 2.04 | 20.03 ± 1.30 | 15% |
| 3 iterations | 4 | 18.60 ± 1.85 | 16.29 ± 1.19 | 17% |
| 5 iterations | 0 | 12.05 ± 0.33 | 10.44 ± 0.52 | 29% |
| 5 iterations | 1 | 11.12 ± 0.35 | 6.67 ± 0.45 | 34% |
| 5 iterations | 2 | 9.19 ± 0.44 | 3.44 ± 0.41 | 44% |
| 5 iterations | 3 | 8.51 ± 0.46 | 2.50 ± 0.62 | 58% |
| 5 iterations | 4 | 7.53 ± 0.50 | 2.03 ± 0.56 | 73% |
| 8 iterations | 0 | 13.01 ± 1.19 | 10.30 ± 0.38 | 28% |
| 8 iterations | 1 | 12.48 ± 1.41 | 6.62 ± 0.44 | 34% |
| 8 iterations | 2 | 10.05 ± 1.30 | 4.09 ± 0.67 | 46% |
| 8 iterations | 3 | 8.85 ± 1.25 | 3.03 ± 0.75 | 58% |
| 8 iterations | 4 | 7.82 ± 1.21 | 2.36 ± 0.73 | 68% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 1095.18 ± 5.06 | 368.02 ± 1.19 | 1% |
| 0 iterations | 1 | 978.91 ± 4.34 | 349.69 ± 1.08 | 1% |
| 0 iterations | 2 | 881.94 ± 4.05 | 326.12 ± 0.95 | 1% |
| 0 iterations | 3 | 849.41 ± 3.92 | 315.03 ± 0.88 | 1% |
| 0 iterations | 4 | 839.82 ± 3.88 | 309.84 ± 0.85 | 1% |
| 3 iterations | 0 | 52.28 ± 3.50 | 49.82 ± 1.90 | 5% |
| 3 iterations | 1 | 43.16 ± 3.20 | 40.13 ± 1.84 | 11% |
| 3 iterations | 2 | 30.57 ± 2.70 | 27.54 ± 1.64 | 18% |
| 3 iterations | 3 | 23.05 ± 2.12 | 19.71 ± 1.30 | 17% |
| 3 iterations | 4 | 19.06 ± 1.89 | 15.92 ± 1.16 | 18% |
| 5 iterations | 0 | 14.72 ± 1.11 | 10.86 ± 0.44 | 29% |
| 5 iterations | 1 | 13.07 ± 0.96 | 6.83 ± 0.41 | 35% |
| 5 iterations | 2 | 12.44 ± 1.32 | 3.85 ± 0.37 | 41% |
| 5 iterations | 3 | 11.06 ± 1.12 | 3.14 ± 0.57 | 56% |
| 5 iterations | 4 | 10.46 ± 1.14 | 2.71 ± 0.58 | 64% |
| 8 iterations | 0 | 15.06 ± 1.29 | 10.60 ± 0.30 | 29% |
| 8 iterations | 1 | 13.93 ± 1.21 | 6.92 ± 0.38 | 37% |
| 8 iterations | 2 | 12.85 ± 1.40 | 4.72 ± 0.59 | 45% |
| 8 iterations | 3 | 11.14 ± 1.18 | 3.88 ± 0.69 | 55% |
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
| (10,12) 12 iterations | central | 2.3 | 4.6 | 7.3 |
| (10,12) 12 iterations | right wing | 3.2 | 6.1 | 8.9 |
| particle method | left wing | 10.2 | 17.3 | 24.3 |
| particle method | central | 2.2 | 4.4 | 6.6 |
| particle method | right wing | 3.1 | 6.2 | 9.3 |
