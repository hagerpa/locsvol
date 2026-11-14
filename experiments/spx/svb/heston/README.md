# svb, Heston backbone

Failure of Silicon Valley Bank; front-end rates collapse, VIX 26. The Heston backbone is fitted to the S&P 500 index options of 2023-02-13 (kappa = 1.7573, theta = 0.0615, sigma = 0.8130, rho = -0.7362, v0 = 0.0391); the leverage factor is calibrated to the 2,117 quotes of 2023-03-13, 1,174 of them liquid, at 14 expiries from 11 to 368 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 8.26 ± 5.38 | 3.03 ± 1.14 | 5.76 ± 0.40 |
| Joint regression (6,8), 5 iterations | 9.64 ± 5.80 | 3.46 ± 0.71 | 14.98 ± 0.40 |
| Particle method | 21.27 ± 7.84 | 10.17 ± 1.17 | 17.09 ± 0.39 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 6.02 | 0.94 | 0.68 |
| Joint regression (6,8), 5 iterations | 6.04 | 0.94 | 0.68 |
| Particle method | 5.77 | 0.93 | 0.65 |

### Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 411.30 ± 2.12 | 312.35 ± 0.70 | 2% |
| (10,12) | 3 iterations | 13.89 ± 1.62 | 4.02 ± 0.27 | 50% |
| (10,12) | 5 iterations | 13.02 ± 1.60 | 4.51 ± 0.32 | 45% |
| (10,12) | 8 iterations | 13.07 ± 1.59 | 4.66 ± 0.31 | 45% |
| (6,8) | 0 iterations | 412.15 ± 2.13 | 312.84 ± 0.70 | 1% |
| (6,8) | 3 iterations | 14.34 ± 1.63 | 5.79 ± 0.16 | 44% |
| (6,8) | 5 iterations | 13.47 ± 1.54 | 6.26 ± 0.26 | 42% |
| (6,8) | 8 iterations | 13.51 ± 1.53 | 6.36 ± 0.26 | 42% |
| Refined SVI target |  | 26.64 | 23.68 | 16% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square change of the leverage factor along the simulated paths in every fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 876.48 ± 1.72 | 291.91 ± 0.35 | 407.23 ± 2.87 | 538.49 ± 1.03 | 379.35 ± 0.53 | 1% | 0% | 99% |
| (10,12) 3 iterations | 11.95 ± 2.70 | 7.62 ± 0.68 | 8.53 ± 2.02 | 26.71 ± 0.36 | 21.80 ± 0.22 | 13% | 4% | 82% |
| (10,12) 5 iterations | 4.46 ± 1.31 | 2.40 ± 0.39 | 4.97 ± 0.69 | 26.67 ± 0.17 | 23.25 ± 0.41 | 15% | 3% | 83% |
| (10,12) 8 iterations | 4.51 ± 1.30 | 2.00 ± 0.21 | 5.01 ± 0.84 | 26.69 ± 0.17 | 23.30 ± 0.40 | 15% | 3% | 83% |
| (10,12) 12 iterations | 4.51 ± 1.31 | 2.00 ± 0.21 | 5.02 ± 0.85 | 26.68 ± 0.17 | 23.30 ± 0.40 | 15% | 3% | 83% |
| (10,12) 0 iterations + 4 correction steps | 671.20 ± 1.77 | 244.29 ± 0.27 | 396.47 ± 3.22 | 409.88 ± 0.94 | 311.57 ± 0.44 | 2% | 0% | 98% |
| (10,12) 3 iterations + 4 correction steps | 26.61 ± 1.87 | 23.62 ± 0.30 | 20.83 ± 0.81 | 15.69 ± 0.76 | 4.02 ± 0.26 | 51% | 13% | 36% |
| (10,12) 5 iterations + 4 correction steps | 24.31 ± 1.60 | 22.74 ± 0.30 | 19.40 ± 0.25 | 14.66 ± 0.70 | 4.39 ± 0.25 | 47% | 13% | 40% |
| (10,12) 8 iterations + 4 correction steps | 24.21 ± 1.60 | 22.67 ± 0.30 | 19.38 ± 0.26 | 14.70 ± 0.70 | 4.53 ± 0.25 | 47% | 13% | 41% |
| (6,8) 3 iterations | 11.73 ± 2.50 | 8.05 ± 0.66 | 16.24 ± 1.65 | 28.81 ± 0.34 | 24.05 ± 0.21 | 13% | 4% | 83% |
| (6,8) 5 iterations | 5.56 ± 1.18 | 2.97 ± 0.27 | 14.36 ± 0.47 | 28.93 ± 0.18 | 25.41 ± 0.37 | 15% | 3% | 82% |
| (6,8) 8 iterations | 5.62 ± 1.20 | 2.68 ± 0.17 | 14.26 ± 0.43 | 28.93 ± 0.18 | 25.45 ± 0.36 | 14% | 4% | 82% |
| (6,8) 12 iterations | 5.62 ± 1.20 | 2.68 ± 0.17 | 14.25 ± 0.43 | 28.93 ± 0.18 | 25.45 ± 0.36 | 14% | 4% | 82% |
| (6,8) 0 iterations + 4 correction steps | 672.52 ± 1.80 | 244.69 ± 0.27 | 396.19 ± 3.17 | 410.79 ± 0.95 | 312.07 ± 0.44 | 2% | 0% | 98% |
| (6,8) 3 iterations + 4 correction steps | 25.39 ± 1.69 | 23.63 ± 0.30 | 19.30 ± 1.27 | 15.78 ± 0.69 | 5.78 ± 0.18 | 49% | 11% | 40% |
| (6,8) 5 iterations + 4 correction steps | 23.15 ± 1.43 | 22.43 ± 0.30 | 17.57 ± 0.34 | 14.73 ± 0.63 | 6.08 ± 0.17 | 45% | 12% | 43% |
| (6,8) 8 iterations + 4 correction steps | 23.03 ± 1.43 | 22.37 ± 0.30 | 17.53 ± 0.28 | 14.77 ± 0.62 | 6.17 ± 0.17 | 45% | 11% | 43% |
| particle method | 17.92 ± 1.89 | 9.54 ± 0.26 | 17.12 ± 0.67 | 27.91 ± 0.28 | 22.69 ± 0.50 | 18% | 3% | 79% |
| refined SVI target |  |  |  | 26.64 | 23.68 | 16% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square change of the leverage factor along the simulated paths in every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 7.44 ± 2.82 | 5.62 ± 0.73 | 3.51 ± 2.60 | 0.03 ± 0.28 | -1.50 ± 0.24 |
| (10,12) 5 iterations | -0.05 ± 0.12 | 0.40 ± 0.28 | -0.05 ± 0.27 | -0.01 ± 0.01 | -0.05 ± 0.02 |
| (10,12) 8 iterations | 0.00 ± 0.00 | -0.00 ± 0.00 | -0.01 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| (6,8) 3 iterations | 6.11 ± 2.84 | 5.37 ± 0.63 | 1.99 ± 1.56 | -0.12 ± 0.26 | -1.40 ± 0.21 |
| (6,8) 5 iterations | -0.06 ± 0.08 | 0.29 ± 0.24 | 0.11 ± 0.14 | 0.00 ± 0.01 | -0.04 ± 0.01 |
| (6,8) 8 iterations | 0.00 ± 0.00 | -0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 538.49 ± 1.03 | 379.35 ± 0.53 | 1% |
| 0 iterations | 1 | 503.10 ± 0.97 | 364.61 ± 0.51 | 1% |
| 0 iterations | 2 | 452.55 ± 0.94 | 340.43 ± 0.48 | 1% |
| 0 iterations | 3 | 423.98 ± 0.93 | 322.66 ± 0.46 | 1% |
| 0 iterations | 4 | 409.88 ± 0.94 | 311.57 ± 0.44 | 2% |
| 3 iterations | 0 | 26.71 ± 0.36 | 21.80 ± 0.22 | 13% |
| 3 iterations | 1 | 23.35 ± 0.50 | 15.25 ± 0.21 | 17% |
| 3 iterations | 2 | 19.02 ± 0.70 | 8.76 ± 0.22 | 24% |
| 3 iterations | 3 | 17.06 ± 0.76 | 5.74 ± 0.27 | 39% |
| 3 iterations | 4 | 15.69 ± 0.76 | 4.02 ± 0.26 | 51% |
| 5 iterations | 0 | 26.67 ± 0.17 | 23.25 ± 0.41 | 15% |
| 5 iterations | 1 | 23.27 ± 0.32 | 16.34 ± 0.32 | 17% |
| 5 iterations | 2 | 18.60 ± 0.59 | 9.44 ± 0.21 | 22% |
| 5 iterations | 3 | 16.19 ± 0.68 | 6.26 ± 0.21 | 34% |
| 5 iterations | 4 | 14.66 ± 0.70 | 4.39 ± 0.25 | 47% |
| 8 iterations | 0 | 26.69 ± 0.17 | 23.30 ± 0.40 | 15% |
| 8 iterations | 1 | 23.29 ± 0.32 | 16.46 ± 0.31 | 17% |
| 8 iterations | 2 | 18.63 ± 0.59 | 9.62 ± 0.21 | 22% |
| 8 iterations | 3 | 16.22 ± 0.68 | 6.43 ± 0.22 | 34% |
| 8 iterations | 4 | 14.70 ± 0.70 | 4.53 ± 0.25 | 47% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 538.49 ± 1.03 | 379.35 ± 0.53 | 1% |
| 0 iterations | 1 | 502.88 ± 0.98 | 364.66 ± 0.51 | 1% |
| 0 iterations | 2 | 452.58 ± 0.94 | 340.51 ± 0.48 | 1% |
| 0 iterations | 3 | 424.25 ± 0.94 | 322.84 ± 0.46 | 1% |
| 0 iterations | 4 | 410.79 ± 0.95 | 312.07 ± 0.44 | 2% |
| 3 iterations | 0 | 28.81 ± 0.34 | 24.05 ± 0.21 | 13% |
| 3 iterations | 1 | 24.69 ± 0.45 | 17.22 ± 0.20 | 17% |
| 3 iterations | 2 | 19.22 ± 0.63 | 9.97 ± 0.21 | 26% |
| 3 iterations | 3 | 17.20 ± 0.68 | 7.21 ± 0.20 | 38% |
| 3 iterations | 4 | 15.78 ± 0.69 | 5.78 ± 0.18 | 49% |
| 5 iterations | 0 | 28.93 ± 0.18 | 25.41 ± 0.37 | 15% |
| 5 iterations | 1 | 24.77 ± 0.28 | 18.33 ± 0.31 | 17% |
| 5 iterations | 2 | 18.91 ± 0.52 | 10.65 ± 0.21 | 24% |
| 5 iterations | 3 | 16.34 ± 0.60 | 7.63 ± 0.18 | 36% |
| 5 iterations | 4 | 14.73 ± 0.63 | 6.08 ± 0.17 | 45% |
| 8 iterations | 0 | 28.93 ± 0.18 | 25.45 ± 0.36 | 14% |
| 8 iterations | 1 | 24.78 ± 0.28 | 18.42 ± 0.30 | 17% |
| 8 iterations | 2 | 18.94 ± 0.52 | 10.80 ± 0.21 | 24% |
| 8 iterations | 3 | 16.37 ± 0.60 | 7.77 ± 0.18 | 35% |
| 8 iterations | 4 | 14.77 ± 0.62 | 6.17 ± 0.17 | 45% |

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
| starting point L0 | left wing | 2.8 | 4.0 | 6.8 |
| starting point L0 | central | 0.9 | 1.5 | 2.3 |
| starting point L0 | right wing | 3.2 | 11.7 | 34.9 |
| (10,12) 12 iterations | left wing | 4.0 | 6.4 | 11.5 |
| (10,12) 12 iterations | central | 1.3 | 2.3 | 3.2 |
| (10,12) 12 iterations | right wing | 2.4 | 6.8 | 12.1 |
| particle method | left wing | 3.9 | 6.3 | 11.5 |
| particle method | central | 1.3 | 2.2 | 3.2 |
| particle method | right wing | 2.4 | 6.0 | 9.4 |
