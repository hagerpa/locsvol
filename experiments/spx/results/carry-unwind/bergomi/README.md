# carry-unwind, Bergomi backbone

Unwind of the yen carry trade; VIX intraday above 60, SPX about -3%. The Bergomi backbone is fitted to the S&P 500 index options of 2024-07-08 (nu = 2.3009, theta = 0.1308, rho1 = -0.6530, rho2 = -0.3701); the leverage factor is calibrated to the 2,739 quotes of 2024-08-05, 1,947 of them liquid, at 15 expiries from 11 to 375 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 5.26 ± 3.56 | 4.17 ± 0.91 | 5.65 ± 0.36 |
| Joint regression (6,8), 5 iterations | 6.29 ± 3.26 | 3.37 ± 0.65 | 4.78 ± 0.33 |
| Particle method | 32.26 ± 4.28 | 23.27 ± 1.19 | 12.23 ± 0.31 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 3.09 | 0.90 | 0.41 |
| Joint regression (6,8), 5 iterations | 3.09 | 0.90 | 0.41 |
| Particle method | 3.05 | 0.89 | 0.41 |

### Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 325.13 ± 1.43 | 263.27 ± 0.73 | 5% |
| (10,12) | 3 iterations | 6.81 ± 0.86 | 0.27 ± 0.34 | 97% |
| (10,12) | 5 iterations | 5.23 ± 0.71 | 0.25 ± 0.31 | 98% |
| (10,12) | 8 iterations | 4.97 ± 0.67 | 0.26 ± 0.33 | 98% |
| (6,8) | 0 iterations | 326.28 ± 1.43 | 264.16 ± 0.74 | 5% |
| (6,8) | 3 iterations | 6.30 ± 0.79 | 0.22 ± 0.36 | 97% |
| (6,8) | 5 iterations | 5.11 ± 0.70 | 0.30 ± 0.36 | 97% |
| (6,8) | 8 iterations | 5.06 ± 0.71 | 0.27 ± 0.33 | 98% |
| Particle method |  | 17.86 ± 1.02 | 18.25 ± 1.06 | 66% |
| Refined SVI target |  | 9.89 | 8.06 | 77% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square coefficient increment of the fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 821.81 ± 2.73 | 264.16 ± 0.64 | 352.67 ± 0.96 | 446.61 ± 1.40 | 351.40 ± 0.79 | 5% | 0% | 95% |
| (10,12) 3 iterations | 16.72 ± 2.72 | 6.62 ± 0.44 | 7.52 ± 0.46 | 11.44 ± 0.44 | 7.13 ± 0.12 | 80% | 1% | 19% |
| (10,12) 5 iterations | 6.39 ± 2.31 | 4.15 ± 0.58 | 5.53 ± 0.28 | 10.15 ± 0.33 | 7.19 ± 0.14 | 79% | 2% | 20% |
| (10,12) 8 iterations | 6.31 ± 2.29 | 4.06 ± 0.50 | 5.38 ± 0.28 | 10.10 ± 0.33 | 7.19 ± 0.14 | 79% | 2% | 20% |
| (10,12) 12 iterations | 6.32 ± 2.29 | 4.06 ± 0.50 | 5.36 ± 0.28 | 10.10 ± 0.33 | 7.19 ± 0.14 | 79% | 2% | 20% |
| (10,12) 0 iterations + 4 correction steps | 585.49 ± 2.81 | 185.19 ± 0.41 | 347.29 ± 0.98 | 323.79 ± 1.27 | 262.59 ± 0.64 | 5% | 0% | 95% |
| (10,12) 3 iterations + 4 correction steps | 15.37 ± 2.27 | 14.00 ± 0.28 | 12.87 ± 0.24 | 7.33 ± 0.54 | 0.33 ± 0.16 | 97% | 2% | 1% |
| (10,12) 5 iterations + 4 correction steps | 11.10 ± 1.22 | 13.69 ± 0.31 | 12.30 ± 0.16 | 5.65 ± 0.44 | 0.26 ± 0.14 | 98% | 2% | 1% |
| (10,12) 8 iterations + 4 correction steps | 11.34 ± 1.13 | 13.65 ± 0.31 | 12.33 ± 0.16 | 5.38 ± 0.42 | 0.30 ± 0.15 | 97% | 2% | 1% |
| (6,8) 3 iterations | 15.59 ± 2.66 | 6.57 ± 0.53 | 7.05 ± 0.42 | 11.54 ± 0.43 | 7.48 ± 0.14 | 81% | 1% | 18% |
| (6,8) 5 iterations | 6.78 ± 2.07 | 3.58 ± 0.46 | 4.72 ± 0.35 | 10.54 ± 0.34 | 7.57 ± 0.15 | 80% | 2% | 19% |
| (6,8) 8 iterations | 6.66 ± 2.04 | 3.62 ± 0.44 | 4.62 ± 0.34 | 10.49 ± 0.34 | 7.57 ± 0.15 | 80% | 2% | 19% |
| (6,8) 12 iterations | 6.67 ± 2.04 | 3.62 ± 0.44 | 4.61 ± 0.34 | 10.49 ± 0.34 | 7.58 ± 0.15 | 80% | 2% | 19% |
| (6,8) 0 iterations + 4 correction steps | 587.74 ± 2.81 | 185.84 ± 0.41 | 347.49 ± 0.99 | 324.90 ± 1.27 | 263.46 ± 0.64 | 5% | 0% | 95% |
| (6,8) 3 iterations + 4 correction steps | 13.75 ± 2.11 | 14.52 ± 0.28 | 13.49 ± 0.22 | 6.78 ± 0.52 | 0.31 ± 0.19 | 98% | 1% | 1% |
| (6,8) 5 iterations + 4 correction steps | 10.88 ± 1.20 | 13.88 ± 0.30 | 12.64 ± 0.18 | 5.55 ± 0.46 | 0.30 ± 0.18 | 98% | 1% | 1% |
| (6,8) 8 iterations + 4 correction steps | 10.75 ± 1.09 | 13.91 ± 0.30 | 12.60 ± 0.18 | 5.50 ± 0.45 | 0.22 ± 0.15 | 98% | 2% | 0% |
| particle method | 31.08 ± 2.04 | 23.37 ± 0.80 | 12.17 ± 0.26 | 17.86 ± 0.42 | 18.22 ± 0.50 | 67% | 3% | 31% |
| refined SVI target |  |  |  | 9.89 | 8.06 | 77% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square coefficient increment of every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 10.41 ± 1.11 | 2.57 ± 0.44 | 2.16 ± 0.41 | 1.34 ± 0.15 | -0.06 ± 0.04 |
| (10,12) 5 iterations | 0.07 ± 0.08 | 0.09 ± 0.17 | 0.17 ± 0.03 | 0.05 ± 0.01 | -0.00 ± 0.00 |
| (10,12) 8 iterations | -0.01 ± 0.00 | -0.00 ± 0.00 | 0.02 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 |
| (6,8) 3 iterations | 8.92 ± 1.58 | 2.95 ± 0.34 | 2.44 ± 0.30 | 1.05 ± 0.13 | -0.09 ± 0.03 |
| (6,8) 5 iterations | 0.11 ± 0.13 | -0.04 ± 0.21 | 0.11 ± 0.03 | 0.04 ± 0.01 | -0.01 ± 0.00 |
| (6,8) 8 iterations | -0.00 ± 0.00 | -0.00 ± 0.00 | 0.01 ± 0.00 | -0.00 ± 0.00 | -0.00 ± 0.00 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 446.61 ± 1.40 | 351.40 ± 0.79 | 5% |
| 0 iterations | 1 | 402.53 ± 1.38 | 320.96 ± 0.74 | 5% |
| 0 iterations | 2 | 354.58 ± 1.33 | 286.72 ± 0.67 | 4% |
| 0 iterations | 3 | 335.13 ± 1.30 | 272.15 ± 0.65 | 4% |
| 0 iterations | 4 | 323.79 ± 1.27 | 262.59 ± 0.64 | 5% |
| 3 iterations | 0 | 11.44 ± 0.44 | 7.13 ± 0.12 | 80% |
| 3 iterations | 1 | 9.60 ± 0.54 | 2.26 ± 0.13 | 89% |
| 3 iterations | 2 | 7.41 ± 0.55 | 0.35 ± 0.17 | 97% |
| 3 iterations | 3 | 7.33 ± 0.54 | 0.33 ± 0.16 | 97% |
| 3 iterations | 4 | 7.33 ± 0.54 | 0.33 ± 0.16 | 97% |
| 5 iterations | 0 | 10.15 ± 0.33 | 7.19 ± 0.14 | 79% |
| 5 iterations | 1 | 8.22 ± 0.46 | 2.33 ± 0.16 | 87% |
| 5 iterations | 2 | 5.76 ± 0.45 | 0.33 ± 0.17 | 97% |
| 5 iterations | 3 | 5.65 ± 0.44 | 0.26 ± 0.14 | 98% |
| 5 iterations | 4 | 5.65 ± 0.44 | 0.26 ± 0.14 | 98% |
| 8 iterations | 0 | 10.10 ± 0.33 | 7.19 ± 0.14 | 79% |
| 8 iterations | 1 | 8.15 ± 0.46 | 2.32 ± 0.16 | 87% |
| 8 iterations | 2 | 5.60 ± 0.43 | 0.33 ± 0.17 | 97% |
| 8 iterations | 3 | 5.38 ± 0.42 | 0.30 ± 0.15 | 97% |
| 8 iterations | 4 | 5.38 ± 0.42 | 0.30 ± 0.15 | 97% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 446.61 ± 1.40 | 351.40 ± 0.79 | 5% |
| 0 iterations | 1 | 402.85 ± 1.37 | 321.16 ± 0.74 | 5% |
| 0 iterations | 2 | 354.86 ± 1.33 | 287.06 ± 0.67 | 4% |
| 0 iterations | 3 | 336.03 ± 1.30 | 272.94 ± 0.65 | 4% |
| 0 iterations | 4 | 324.90 ± 1.27 | 263.46 ± 0.64 | 5% |
| 3 iterations | 0 | 11.54 ± 0.43 | 7.48 ± 0.14 | 81% |
| 3 iterations | 1 | 9.49 ± 0.54 | 2.42 ± 0.14 | 88% |
| 3 iterations | 2 | 6.90 ± 0.52 | 0.26 ± 0.17 | 98% |
| 3 iterations | 3 | 6.78 ± 0.52 | 0.31 ± 0.19 | 98% |
| 3 iterations | 4 | 6.78 ± 0.52 | 0.31 ± 0.19 | 98% |
| 5 iterations | 0 | 10.54 ± 0.34 | 7.57 ± 0.15 | 80% |
| 5 iterations | 1 | 8.44 ± 0.49 | 2.50 ± 0.17 | 86% |
| 5 iterations | 2 | 5.65 ± 0.47 | 0.18 ± 0.11 | 99% |
| 5 iterations | 3 | 5.55 ± 0.46 | 0.30 ± 0.18 | 98% |
| 5 iterations | 4 | 5.55 ± 0.46 | 0.30 ± 0.18 | 98% |
| 8 iterations | 0 | 10.49 ± 0.34 | 7.57 ± 0.15 | 80% |
| 8 iterations | 1 | 8.37 ± 0.49 | 2.50 ± 0.17 | 86% |
| 8 iterations | 2 | 5.63 ± 0.46 | 0.25 ± 0.16 | 98% |
| 8 iterations | 3 | 5.50 ± 0.45 | 0.22 ± 0.15 | 98% |
| 8 iterations | 4 | 5.50 ± 0.45 | 0.22 ± 0.15 | 98% |

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
| starting point L0 | left wing | 3.6 | 6.1 | 11.8 |
| starting point L0 | central | 1.3 | 1.9 | 2.5 |
| starting point L0 | right wing | 1.3 | 3.8 | 13.3 |
| (10,12) 12 iterations | left wing | 3.5 | 5.7 | 13.1 |
| (10,12) 12 iterations | central | 1.5 | 2.2 | 2.7 |
| (10,12) 12 iterations | right wing | 1.0 | 1.9 | 4.9 |
| particle method | left wing | 3.6 | 5.6 | 13.2 |
| particle method | central | 1.6 | 2.3 | 2.8 |
| particle method | right wing | 1.0 | 1.8 | 4.4 |
