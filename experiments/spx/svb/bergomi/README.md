# svb, Bergomi backbone

Failure of Silicon Valley Bank; front-end rates collapse, VIX 26. The Bergomi backbone is fitted to the S&P 500 index options of 2023-02-13 (nu = 2.0667, theta = 0.1050, rho1 = -0.6458, rho2 = -0.7380); the leverage factor is calibrated to the 2,117 quotes of 2023-03-13, 1,174 of them liquid, at 14 expiries from 11 to 368 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 11.54 ± 9.20 | 3.77 ± 1.09 | 7.12 ± 0.66 |
| Joint regression (6,8), 5 iterations | 10.15 ± 8.53 | 3.10 ± 0.73 | 10.43 ± 0.49 |
| Particle method | 32.84 ± 8.85 | 11.09 ± 1.17 | 15.51 ± 0.51 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 5.49 | 0.91 | 0.69 |
| Joint regression (6,8), 5 iterations | 5.52 | 0.91 | 0.68 |
| Particle method | 5.28 | 0.90 | 0.66 |

### Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 400.08 ± 2.33 | 295.50 ± 0.67 | 2% |
| (10,12) | 3 iterations | 14.78 ± 1.15 | 3.41 ± 0.71 | 53% |
| (10,12) | 5 iterations | 12.31 ± 0.73 | 4.71 ± 0.89 | 42% |
| (10,12) | 8 iterations | 12.22 ± 0.73 | 4.70 ± 0.89 | 42% |
| (6,8) | 0 iterations | 401.17 ± 2.34 | 296.02 ± 0.67 | 2% |
| (6,8) | 3 iterations | 15.34 ± 1.04 | 4.90 ± 0.61 | 51% |
| (6,8) | 5 iterations | 12.87 ± 0.69 | 5.93 ± 0.77 | 44% |
| (6,8) | 8 iterations | 12.73 ± 0.68 | 5.91 ± 0.77 | 44% |
| Particle method |  | 30.02 ± 1.41 | 23.80 ± 0.78 | 16% |
| Refined SVI target |  | 26.64 | 23.68 | 16% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square coefficient increment of the fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 841.66 ± 1.90 | 274.33 ± 0.30 | 420.92 ± 4.17 | 518.38 ± 1.17 | 355.55 ± 0.50 | 1% | 0% | 98% |
| (10,12) 3 iterations | 16.28 ± 2.65 | 6.61 ± 0.21 | 20.53 ± 2.63 | 27.57 ± 0.41 | 21.58 ± 0.28 | 16% | 3% | 81% |
| (10,12) 5 iterations | 4.10 ± 1.73 | 2.64 ± 0.42 | 8.04 ± 2.27 | 26.72 ± 0.20 | 23.63 ± 0.40 | 14% | 3% | 83% |
| (10,12) 8 iterations | 4.16 ± 1.71 | 2.51 ± 0.29 | 7.23 ± 2.05 | 26.60 ± 0.20 | 23.52 ± 0.39 | 14% | 3% | 84% |
| (10,12) 12 iterations | 4.15 ± 1.71 | 2.51 ± 0.29 | 7.22 ± 2.04 | 26.60 ± 0.20 | 23.52 ± 0.39 | 14% | 3% | 84% |
| (10,12) 0 iterations + 4 correction steps | 651.26 ± 1.83 | 234.38 ± 0.26 | 403.65 ± 4.31 | 397.48 ± 0.95 | 294.97 ± 0.38 | 2% | 0% | 98% |
| (10,12) 3 iterations + 4 correction steps | 26.78 ± 1.78 | 23.46 ± 0.23 | 28.07 ± 1.88 | 15.71 ± 0.98 | 2.94 ± 0.32 | 57% | 12% | 31% |
| (10,12) 5 iterations + 4 correction steps | 20.64 ± 1.13 | 22.15 ± 0.23 | 21.87 ± 0.96 | 13.19 ± 0.73 | 4.10 ± 0.49 | 42% | 13% | 45% |
| (10,12) 8 iterations + 4 correction steps | 20.71 ± 1.10 | 22.18 ± 0.23 | 21.60 ± 0.81 | 13.09 ± 0.72 | 4.10 ± 0.48 | 42% | 13% | 45% |
| (6,8) 3 iterations | 16.87 ± 2.69 | 6.83 ± 0.23 | 23.44 ± 2.14 | 28.46 ± 0.39 | 22.92 ± 0.26 | 15% | 3% | 81% |
| (6,8) 5 iterations | 6.70 ± 1.92 | 2.99 ± 0.30 | 10.96 ± 1.50 | 27.68 ± 0.18 | 24.85 ± 0.38 | 14% | 2% | 84% |
| (6,8) 8 iterations | 6.82 ± 1.89 | 2.91 ± 0.22 | 10.44 ± 1.26 | 27.54 ± 0.19 | 24.73 ± 0.37 | 14% | 3% | 84% |
| (6,8) 12 iterations | 6.81 ± 1.89 | 2.91 ± 0.22 | 10.43 ± 1.26 | 27.54 ± 0.19 | 24.73 ± 0.37 | 14% | 3% | 84% |
| (6,8) 0 iterations + 4 correction steps | 652.55 ± 1.83 | 234.74 ± 0.26 | 404.46 ± 4.37 | 398.47 ± 0.94 | 295.47 ± 0.38 | 2% | 0% | 98% |
| (6,8) 3 iterations + 4 correction steps | 24.33 ± 1.98 | 23.31 ± 0.21 | 27.14 ± 1.73 | 16.09 ± 0.91 | 4.43 ± 0.29 | 53% | 10% | 36% |
| (6,8) 5 iterations + 4 correction steps | 19.95 ± 0.97 | 22.27 ± 0.22 | 19.63 ± 0.88 | 13.52 ± 0.65 | 5.40 ± 0.44 | 45% | 9% | 47% |
| (6,8) 8 iterations + 4 correction steps | 20.05 ± 0.93 | 22.31 ± 0.22 | 19.40 ± 0.72 | 13.39 ± 0.64 | 5.38 ± 0.44 | 45% | 9% | 46% |
| particle method | 21.24 ± 2.74 | 9.58 ± 0.50 | 16.23 ± 1.28 | 28.51 ± 0.49 | 23.32 ± 0.52 | 16% | 3% | 81% |
| refined SVI target |  |  |  | 26.64 | 23.68 | 16% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square coefficient increment of every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 12.13 ± 3.24 | 4.10 ± 0.28 | 13.31 ± 0.97 | 0.97 ± 0.33 | -1.94 ± 0.15 |
| (10,12) 5 iterations | -0.05 ± 0.05 | 0.13 ± 0.17 | 0.82 ± 0.33 | 0.12 ± 0.01 | 0.11 ± 0.01 |
| (10,12) 8 iterations | 0.01 ± 0.01 | -0.00 ± 0.00 | 0.01 ± 0.01 | -0.00 ± 0.00 | 0.00 ± 0.00 |
| (6,8) 3 iterations | 10.06 ± 2.96 | 3.92 ± 0.20 | 13.01 ± 1.08 | 0.92 ± 0.32 | -1.81 ± 0.15 |
| (6,8) 5 iterations | -0.11 ± 0.08 | 0.08 ± 0.16 | 0.53 ± 0.29 | 0.14 ± 0.01 | 0.12 ± 0.01 |
| (6,8) 8 iterations | 0.01 ± 0.01 | -0.00 ± 0.00 | 0.01 ± 0.01 | -0.00 ± 0.00 | 0.00 ± 0.00 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 518.38 ± 1.17 | 355.55 ± 0.50 | 1% |
| 0 iterations | 1 | 481.60 ± 1.11 | 342.14 ± 0.47 | 1% |
| 0 iterations | 2 | 433.32 ± 1.03 | 320.33 ± 0.43 | 2% |
| 0 iterations | 3 | 409.62 ± 0.97 | 304.95 ± 0.39 | 2% |
| 0 iterations | 4 | 397.48 ± 0.95 | 294.97 ± 0.38 | 2% |
| 3 iterations | 0 | 27.57 ± 0.41 | 21.58 ± 0.28 | 16% |
| 3 iterations | 1 | 24.10 ± 0.55 | 14.72 ± 0.32 | 18% |
| 3 iterations | 2 | 19.53 ± 0.82 | 7.64 ± 0.34 | 26% |
| 3 iterations | 3 | 17.26 ± 0.90 | 4.50 ± 0.39 | 40% |
| 3 iterations | 4 | 15.71 ± 0.98 | 2.94 ± 0.32 | 57% |
| 5 iterations | 0 | 26.72 ± 0.20 | 23.63 ± 0.40 | 14% |
| 5 iterations | 1 | 22.97 ± 0.29 | 16.74 ± 0.42 | 17% |
| 5 iterations | 2 | 17.73 ± 0.55 | 9.61 ± 0.45 | 22% |
| 5 iterations | 3 | 14.80 ± 0.64 | 6.12 ± 0.50 | 32% |
| 5 iterations | 4 | 13.19 ± 0.73 | 4.10 ± 0.49 | 42% |
| 8 iterations | 0 | 26.60 ± 0.20 | 23.52 ± 0.39 | 14% |
| 8 iterations | 1 | 22.84 ± 0.29 | 16.67 ± 0.41 | 16% |
| 8 iterations | 2 | 17.59 ± 0.55 | 9.57 ± 0.44 | 22% |
| 8 iterations | 3 | 14.68 ± 0.63 | 6.10 ± 0.49 | 32% |
| 8 iterations | 4 | 13.09 ± 0.72 | 4.10 ± 0.48 | 42% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 518.38 ± 1.17 | 355.55 ± 0.50 | 1% |
| 0 iterations | 1 | 481.75 ± 1.12 | 342.19 ± 0.47 | 1% |
| 0 iterations | 2 | 433.46 ± 1.04 | 320.46 ± 0.43 | 2% |
| 0 iterations | 3 | 410.03 ± 0.97 | 305.18 ± 0.40 | 2% |
| 0 iterations | 4 | 398.47 ± 0.94 | 295.47 ± 0.38 | 2% |
| 3 iterations | 0 | 28.46 ± 0.39 | 22.92 ± 0.26 | 15% |
| 3 iterations | 1 | 24.76 ± 0.52 | 15.88 ± 0.30 | 18% |
| 3 iterations | 2 | 19.82 ± 0.83 | 8.48 ± 0.31 | 27% |
| 3 iterations | 3 | 17.73 ± 0.88 | 5.70 ± 0.33 | 43% |
| 3 iterations | 4 | 16.09 ± 0.91 | 4.43 ± 0.29 | 53% |
| 5 iterations | 0 | 27.68 ± 0.18 | 24.85 ± 0.38 | 14% |
| 5 iterations | 1 | 23.68 ± 0.27 | 17.79 ± 0.41 | 15% |
| 5 iterations | 2 | 18.01 ± 0.56 | 10.36 ± 0.43 | 23% |
| 5 iterations | 3 | 15.29 ± 0.64 | 7.07 ± 0.46 | 34% |
| 5 iterations | 4 | 13.52 ± 0.65 | 5.40 ± 0.44 | 45% |
| 8 iterations | 0 | 27.54 ± 0.19 | 24.73 ± 0.37 | 14% |
| 8 iterations | 1 | 23.52 ± 0.27 | 17.71 ± 0.40 | 15% |
| 8 iterations | 2 | 17.83 ± 0.56 | 10.31 ± 0.42 | 23% |
| 8 iterations | 3 | 15.14 ± 0.63 | 7.04 ± 0.46 | 34% |
| 8 iterations | 4 | 13.39 ± 0.64 | 5.38 ± 0.44 | 45% |

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
| starting point L0 | left wing | 3.0 | 4.6 | 7.8 |
| starting point L0 | central | 0.9 | 1.6 | 2.7 |
| starting point L0 | right wing | 4.5 | 14.9 | 25.9 |
| (10,12) 12 iterations | left wing | 3.6 | 6.3 | 12.5 |
| (10,12) 12 iterations | central | 1.2 | 1.8 | 2.9 |
| (10,12) 12 iterations | right wing | 2.7 | 7.4 | 13.4 |
| particle method | left wing | 3.6 | 6.4 | 13.1 |
| particle method | central | 1.2 | 1.8 | 3.1 |
| particle method | right wing | 2.7 | 6.6 | 11.4 |
