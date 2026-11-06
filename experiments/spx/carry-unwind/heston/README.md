# carry-unwind, Heston backbone

Unwind of the yen carry trade; VIX intraday above 60, SPX about -3%. The Heston backbone is fitted to the S&P 500 index options of 2024-07-08 (kappa = 3.8048, theta = 0.0358, sigma = 1.1183, rho = -0.6559, v0 = 0.0107); the leverage factor is calibrated to the 2,739 quotes of 2024-08-05, 1,947 of them liquid, at 15 expiries from 11 to 375 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 4.79 ± 3.39 | 6.85 ± 1.24 | 9.48 ± 0.44 |
| Joint regression (6,8), 5 iterations | 7.50 ± 2.44 | 7.12 ± 1.16 | 6.64 ± 0.35 |
| Particle method | 56.72 ± 4.08 | 35.08 ± 1.16 | 21.64 ± 0.29 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 3.02 | 0.89 | 0.39 |
| Joint regression (6,8), 5 iterations | 3.03 | 0.89 | 0.39 |
| Particle method | 2.95 | 0.89 | 0.40 |

### Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 439.09 ± 1.43 | 345.34 ± 0.75 | 3% |
| (10,12) | 3 iterations | 8.01 ± 0.78 | 2.45 ± 0.62 | 96% |
| (10,12) | 5 iterations | 5.54 ± 0.60 | 2.38 ± 0.61 | 96% |
| (10,12) | 8 iterations | 5.18 ± 0.56 | 2.39 ± 0.61 | 95% |
| (6,8) | 0 iterations | 466.64 ± 1.45 | 366.63 ± 0.78 | 3% |
| (6,8) | 3 iterations | 7.65 ± 0.69 | 2.38 ± 0.62 | 96% |
| (6,8) | 5 iterations | 5.51 ± 0.59 | 2.22 ± 0.61 | 95% |
| (6,8) | 8 iterations | 5.26 ± 0.57 | 2.19 ± 0.61 | 95% |
| Refined SVI target |  | 9.89 | 8.06 | 77% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square change of the leverage factor along the simulated paths in every fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 1048.29 ± 2.88 | 340.37 ± 0.59 | 310.98 ± 0.97 | 583.91 ± 1.49 | 443.73 ± 0.83 | 3% | 0% | 96% |
| (10,12) 3 iterations | 39.01 ± 5.20 | 15.76 ± 1.23 | 10.84 ± 0.88 | 15.25 ± 1.27 | 8.53 ± 0.38 | 78% | 3% | 19% |
| (10,12) 5 iterations | 4.98 ± 3.25 | 6.84 ± 1.26 | 9.32 ± 0.72 | 11.06 ± 0.56 | 8.82 ± 0.31 | 73% | 6% | 21% |
| (10,12) 8 iterations | 4.11 ± 2.16 | 4.34 ± 1.11 | 9.97 ± 0.78 | 10.90 ± 0.51 | 8.87 ± 0.30 | 73% | 6% | 20% |
| (10,12) 12 iterations | 4.08 ± 2.28 | 4.26 ± 1.10 | 9.99 ± 0.78 | 10.89 ± 0.51 | 8.86 ± 0.31 | 74% | 6% | 20% |
| (10,12) 0 iterations + 4 correction steps | 793.49 ± 3.11 | 255.77 ± 0.41 | 319.93 ± 0.99 | 438.44 ± 1.38 | 344.87 ± 0.69 | 3% | 0% | 96% |
| (10,12) 3 iterations + 4 correction steps | 23.55 ± 4.35 | 16.74 ± 0.42 | 14.65 ± 0.64 | 8.04 ± 1.15 | 2.19 ± 0.59 | 96% | 1% | 3% |
| (10,12) 5 iterations + 4 correction steps | 14.61 ± 2.02 | 14.50 ± 0.38 | 12.42 ± 0.56 | 5.50 ± 0.89 | 2.15 ± 0.61 | 95% | 2% | 3% |
| (10,12) 8 iterations + 4 correction steps | 13.96 ± 1.48 | 14.53 ± 0.39 | 13.03 ± 0.63 | 5.15 ± 0.83 | 2.16 ± 0.60 | 95% | 2% | 3% |
| (6,8) 3 iterations | 37.65 ± 5.31 | 16.15 ± 1.29 | 11.54 ± 0.81 | 15.17 ± 1.26 | 8.22 ± 0.39 | 78% | 4% | 18% |
| (6,8) 5 iterations | 7.39 ± 1.94 | 7.06 ± 1.17 | 6.39 ± 0.68 | 10.74 ± 0.53 | 7.82 ± 0.33 | 73% | 6% | 21% |
| (6,8) 8 iterations | 7.04 ± 1.53 | 4.83 ± 0.82 | 6.65 ± 0.72 | 10.60 ± 0.49 | 7.90 ± 0.32 | 74% | 5% | 21% |
| (6,8) 12 iterations | 7.01 ± 1.54 | 4.78 ± 0.80 | 6.68 ± 0.73 | 10.60 ± 0.50 | 7.89 ± 0.32 | 74% | 5% | 21% |
| (6,8) 0 iterations + 2 correction steps | 840.95 ± 3.21 | 271.38 ± 0.49 | 336.83 ± 0.95 | 465.88 ± 1.48 | 366.11 ± 0.76 | 3% | 0% | 97% |
| (6,8) 3 iterations + 4 correction steps | 20.79 ± 4.12 | 17.82 ± 0.44 | 15.32 ± 0.62 | 7.64 ± 1.09 | 2.13 ± 0.60 | 96% | 2% | 3% |
| (6,8) 5 iterations + 4 correction steps | 13.24 ± 1.86 | 16.46 ± 0.44 | 12.26 ± 0.47 | 5.52 ± 0.89 | 1.99 ± 0.59 | 95% | 2% | 3% |
| (6,8) 8 iterations + 4 correction steps | 12.93 ± 1.32 | 16.37 ± 0.44 | 12.69 ± 0.54 | 5.28 ± 0.85 | 1.96 ± 0.59 | 95% | 2% | 3% |
| particle method | 55.63 ± 3.22 | 34.62 ± 1.02 | 21.37 ± 0.70 | 29.46 ± 1.03 | 29.82 ± 0.93 | 55% | 3% | 42% |
| refined SVI target |  |  |  | 9.89 | 8.06 | 77% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square change of the leverage factor along the simulated paths in every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 34.93 ± 4.38 | 11.49 ± 2.31 | 0.85 ± 1.61 | 4.35 ± 0.81 | -0.34 ± 0.41 |
| (10,12) 5 iterations | 0.90 ± 1.72 | 2.58 ± 0.29 | -0.67 ± 0.11 | 0.17 ± 0.05 | -0.04 ± 0.01 |
| (10,12) 8 iterations | 0.03 ± 0.25 | 0.08 ± 0.02 | -0.02 ± 0.01 | 0.00 ± 0.01 | 0.01 ± 0.01 |
| (6,8) 3 iterations | 30.64 ± 5.17 | 11.37 ± 2.06 | 4.86 ± 1.49 | 4.58 ± 0.81 | 0.33 ± 0.47 |
| (6,8) 5 iterations | 0.38 ± 1.09 | 2.28 ± 0.44 | -0.28 ± 0.11 | 0.14 ± 0.04 | -0.08 ± 0.02 |
| (6,8) 8 iterations | 0.03 ± 0.17 | 0.04 ± 0.03 | -0.03 ± 0.01 | -0.00 ± 0.01 | 0.01 ± 0.01 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 583.91 ± 1.49 | 443.73 ± 0.83 | 3% |
| 0 iterations | 1 | 522.76 ± 1.53 | 404.51 ± 0.84 | 3% |
| 0 iterations | 2 | 465.16 ± 1.50 | 365.63 ± 0.77 | 3% |
| 0 iterations | 3 | 447.47 ± 1.43 | 352.78 ± 0.71 | 3% |
| 0 iterations | 4 | 438.44 ± 1.38 | 344.87 ± 0.69 | 3% |
| 3 iterations | 0 | 15.25 ± 1.27 | 8.53 ± 0.38 | 78% |
| 3 iterations | 1 | 11.97 ± 1.28 | 4.03 ± 0.51 | 85% |
| 3 iterations | 2 | 8.48 ± 1.18 | 2.24 ± 0.59 | 95% |
| 3 iterations | 3 | 8.04 ± 1.15 | 2.19 ± 0.59 | 96% |
| 3 iterations | 4 | 8.04 ± 1.15 | 2.19 ± 0.59 | 96% |
| 5 iterations | 0 | 11.06 ± 0.56 | 8.82 ± 0.31 | 73% |
| 5 iterations | 1 | 8.66 ± 0.73 | 4.60 ± 0.42 | 83% |
| 5 iterations | 2 | 6.46 ± 0.92 | 2.36 ± 0.57 | 93% |
| 5 iterations | 3 | 5.53 ± 0.90 | 1.95 ± 0.60 | 95% |
| 5 iterations | 4 | 5.50 ± 0.89 | 2.15 ± 0.61 | 95% |
| 8 iterations | 0 | 10.90 ± 0.51 | 8.87 ± 0.30 | 73% |
| 8 iterations | 1 | 8.40 ± 0.68 | 4.64 ± 0.41 | 83% |
| 8 iterations | 2 | 6.11 ± 0.86 | 2.36 ± 0.57 | 93% |
| 8 iterations | 3 | 5.20 ± 0.85 | 1.89 ± 0.60 | 95% |
| 8 iterations | 4 | 5.15 ± 0.83 | 2.16 ± 0.60 | 95% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 583.91 ± 1.49 | 443.73 ± 0.83 | 3% |
| 0 iterations | 1 | 523.57 ± 1.52 | 404.87 ± 0.83 | 3% |
| 0 iterations | 2 | 465.88 ± 1.48 | 366.11 ± 0.76 | 3% |
| 3 iterations | 0 | 15.17 ± 1.26 | 8.22 ± 0.39 | 78% |
| 3 iterations | 1 | 12.29 ± 1.27 | 3.66 ± 0.51 | 86% |
| 3 iterations | 2 | 8.11 ± 1.17 | 2.15 ± 0.59 | 94% |
| 3 iterations | 3 | 7.63 ± 1.09 | 2.12 ± 0.60 | 96% |
| 3 iterations | 4 | 7.64 ± 1.09 | 2.13 ± 0.60 | 96% |
| 5 iterations | 0 | 10.74 ± 0.53 | 7.82 ± 0.33 | 73% |
| 5 iterations | 1 | 8.75 ± 0.74 | 3.89 ± 0.43 | 83% |
| 5 iterations | 2 | 6.27 ± 0.96 | 2.12 ± 0.57 | 93% |
| 5 iterations | 3 | 5.62 ± 0.91 | 1.90 ± 0.59 | 95% |
| 5 iterations | 4 | 5.52 ± 0.89 | 1.99 ± 0.59 | 95% |
| 8 iterations | 0 | 10.60 ± 0.49 | 7.90 ± 0.32 | 74% |
| 8 iterations | 1 | 8.55 ± 0.70 | 3.95 ± 0.43 | 83% |
| 8 iterations | 2 | 5.98 ± 0.91 | 2.13 ± 0.57 | 93% |
| 8 iterations | 3 | 5.33 ± 0.86 | 1.85 ± 0.59 | 95% |
| 8 iterations | 4 | 5.28 ± 0.85 | 1.96 ± 0.59 | 95% |

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
| starting point L0 | left wing | 4.5 | 6.7 | 8.4 |
| starting point L0 | central | 1.6 | 2.5 | 4.1 |
| starting point L0 | right wing | 2.0 | 3.7 | 7.1 |
| (10,12) 12 iterations | left wing | 6.4 | 9.6 | 12.8 |
| (10,12) 12 iterations | central | 2.1 | 3.3 | 5.2 |
| (10,12) 12 iterations | right wing | 1.7 | 2.7 | 5.0 |
| particle method | left wing | 6.4 | 9.9 | 12.7 |
| particle method | central | 2.1 | 3.3 | 5.3 |
| particle method | right wing | 1.7 | 2.7 | 4.3 |

## Notes

- The chain of (6,8) after 0 iterations ends after 2 of 4 steps: the next step's optimization did not converge.
