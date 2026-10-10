# tariffs, Bergomi backbone

Second day after the tariff announcement of 2 April 2025; SPX about -6%, VIX 45; the earlier day is six weeks before, since March 2025 was itself agitated. The Bergomi backbone is fitted to the S&P 500 index options of 2025-02-19 (nu = 2.3284, theta = 0.1825, rho1 = -0.7902, rho2 = -0.2884); the leverage factor is calibrated to the 2,577 quotes of 2025-04-04, 1,622 of them liquid, at 14 expiries from 13 to 378 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 4.18 ± 2.12 | 3.25 ± 1.28 | 5.79 ± 0.64 |
| Joint regression (6,8), 5 iterations | 5.23 ± 2.44 | 2.92 ± 0.91 | 6.36 ± 0.60 |
| Particle method | 31.18 ± 4.64 | 17.06 ± 1.18 | 31.59 ± 0.58 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 3.03 | 0.97 | 0.56 |
| Joint regression (6,8), 5 iterations | 3.03 | 0.97 | 0.55 |
| Particle method | 3.00 | 0.96 | 0.56 |

### Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 394.15 ± 1.01 | 356.35 ± 0.56 | 5% |
| (10,12) | 3 iterations | 7.56 ± 0.83 | 0.38 ± 0.20 | 99% |
| (10,12) | 5 iterations | 5.01 ± 0.57 | 0.31 ± 0.21 | 99% |
| (10,12) | 8 iterations | 5.02 ± 0.58 | 0.34 ± 0.21 | 99% |
| (6,8) | 0 iterations | 383.30 ± 1.00 | 346.71 ± 0.56 | 6% |
| (6,8) | 3 iterations | 7.37 ± 0.82 | 0.39 ± 0.20 | 99% |
| (6,8) | 5 iterations | 4.80 ± 0.52 | 0.39 ± 0.20 | 99% |
| (6,8) | 8 iterations | 4.81 ± 0.53 | 0.39 ± 0.21 | 99% |
| Particle method |  | 10.17 ± 0.53 | 6.20 ± 0.95 | 84% |
| Refined SVI target |  | 6.37 | 1.13 | 98% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square coefficient increment of the fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 872.06 ± 2.30 | 294.31 ± 0.42 | 648.13 ± 2.28 | 472.96 ± 0.91 | 417.92 ± 0.64 | 5% | 0% | 95% |
| (10,12) 3 iterations | 15.00 ± 2.21 | 7.49 ± 0.62 | 7.10 ± 0.80 | 7.97 ± 0.40 | 1.19 ± 0.09 | 98% | 0% | 1% |
| (10,12) 5 iterations | 3.35 ± 1.45 | 3.85 ± 0.43 | 4.33 ± 1.35 | 6.06 ± 0.27 | 1.49 ± 0.15 | 98% | 0% | 2% |
| (10,12) 8 iterations | 3.64 ± 1.49 | 3.58 ± 0.38 | 4.38 ± 1.36 | 5.98 ± 0.27 | 1.46 ± 0.14 | 98% | 0% | 2% |
| (10,12) 12 iterations | 3.63 ± 1.49 | 3.58 ± 0.38 | 4.39 ± 1.37 | 5.98 ± 0.27 | 1.46 ± 0.14 | 98% | 0% | 2% |
| (10,12) 0 iterations + 3 correction steps | 671.14 ± 2.11 | 255.16 ± 0.28 | 629.48 ± 1.86 | 393.96 ± 0.69 | 356.12 ± 0.45 | 6% | 0% | 94% |
| (10,12) 3 iterations + 4 correction steps | 14.40 ± 1.88 | 11.13 ± 0.60 | 11.56 ± 1.08 | 7.67 ± 0.40 | 0.42 ± 0.13 | 99% | 0% | 1% |
| (10,12) 5 iterations + 4 correction steps | 9.03 ± 0.76 | 11.14 ± 0.59 | 10.90 ± 1.11 | 5.10 ± 0.26 | 0.35 ± 0.14 | 99% | 1% | 1% |
| (10,12) 8 iterations + 4 correction steps | 8.08 ± 0.59 | 10.74 ± 0.60 | 11.01 ± 1.07 | 5.10 ± 0.27 | 0.38 ± 0.15 | 98% | 1% | 1% |
| (6,8) 3 iterations | 16.54 ± 2.22 | 6.87 ± 0.60 | 6.87 ± 0.63 | 8.13 ± 0.41 | 1.15 ± 0.09 | 98% | 0% | 1% |
| (6,8) 5 iterations | 4.53 ± 1.66 | 3.18 ± 0.42 | 4.97 ± 1.35 | 6.09 ± 0.27 | 1.48 ± 0.15 | 98% | 0% | 2% |
| (6,8) 8 iterations | 4.89 ± 1.64 | 2.88 ± 0.33 | 4.97 ± 1.39 | 5.98 ± 0.27 | 1.43 ± 0.14 | 98% | 1% | 2% |
| (6,8) 12 iterations | 4.88 ± 1.64 | 2.87 ± 0.33 | 4.98 ± 1.40 | 5.98 ± 0.27 | 1.42 ± 0.14 | 98% | 1% | 2% |
| (6,8) 0 iterations + 4 correction steps | 653.31 ± 2.11 | 246.99 ± 0.28 | 613.82 ± 1.54 | 383.16 ± 0.68 | 346.50 ± 0.44 | 6% | 0% | 94% |
| (6,8) 3 iterations + 4 correction steps | 16.08 ± 1.89 | 11.66 ± 0.54 | 9.86 ± 0.95 | 7.46 ± 0.40 | 0.43 ± 0.13 | 99% | 0% | 1% |
| (6,8) 5 iterations + 4 correction steps | 11.12 ± 0.57 | 11.17 ± 0.54 | 10.38 ± 1.16 | 4.88 ± 0.24 | 0.44 ± 0.13 | 98% | 1% | 1% |
| (6,8) 8 iterations + 4 correction steps | 10.72 ± 0.59 | 11.03 ± 0.54 | 10.44 ± 1.18 | 4.88 ± 0.25 | 0.44 ± 0.13 | 99% | 1% | 1% |
| particle method | 31.76 ± 2.31 | 17.51 ± 0.42 | 32.60 ± 1.28 | 10.37 ± 0.33 | 6.23 ± 0.43 | 83% | 3% | 14% |
| refined SVI target |  |  |  | 6.37 | 1.13 | 98% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square coefficient increment of every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 11.37 ± 3.12 | 3.91 ± 0.54 | 2.72 ± 0.66 | 1.99 ± 0.15 | -0.27 ± 0.18 |
| (10,12) 5 iterations | -0.28 ± 0.11 | 0.26 ± 0.20 | -0.06 ± 0.03 | 0.08 ± 0.01 | 0.03 ± 0.01 |
| (10,12) 8 iterations | 0.01 ± 0.00 | 0.00 ± 0.00 | -0.01 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| (6,8) 3 iterations | 11.65 ± 2.90 | 3.99 ± 0.59 | 1.90 ± 0.93 | 2.15 ± 0.17 | -0.28 ± 0.17 |
| (6,8) 5 iterations | -0.35 ± 0.14 | 0.30 ± 0.24 | -0.01 ± 0.07 | 0.11 ± 0.01 | 0.05 ± 0.01 |
| (6,8) 8 iterations | 0.01 ± 0.00 | 0.00 ± 0.00 | -0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 472.96 ± 0.91 | 417.92 ± 0.64 | 5% |
| 0 iterations | 1 | 444.13 ± 0.82 | 396.19 ± 0.57 | 5% |
| 0 iterations | 2 | 411.61 ± 0.72 | 371.47 ± 0.48 | 5% |
| 0 iterations | 3 | 393.96 ± 0.69 | 356.12 ± 0.45 | 6% |
| 3 iterations | 0 | 7.97 ± 0.40 | 1.19 ± 0.09 | 98% |
| 3 iterations | 1 | 7.23 ± 0.36 | 0.64 ± 0.06 | 98% |
| 3 iterations | 2 | 7.70 ± 0.39 | 0.50 ± 0.13 | 98% |
| 3 iterations | 3 | 7.67 ± 0.40 | 0.42 ± 0.13 | 99% |
| 3 iterations | 4 | 7.67 ± 0.40 | 0.42 ± 0.13 | 99% |
| 5 iterations | 0 | 6.06 ± 0.27 | 1.49 ± 0.15 | 98% |
| 5 iterations | 1 | 5.93 ± 0.28 | 1.02 ± 0.13 | 98% |
| 5 iterations | 2 | 5.57 ± 0.28 | 0.62 ± 0.12 | 99% |
| 5 iterations | 3 | 5.30 ± 0.28 | 0.46 ± 0.14 | 98% |
| 5 iterations | 4 | 5.10 ± 0.26 | 0.35 ± 0.14 | 99% |
| 8 iterations | 0 | 5.98 ± 0.27 | 1.46 ± 0.14 | 98% |
| 8 iterations | 1 | 5.86 ± 0.28 | 1.05 ± 0.12 | 98% |
| 8 iterations | 2 | 5.52 ± 0.28 | 0.64 ± 0.11 | 99% |
| 8 iterations | 3 | 5.26 ± 0.27 | 0.47 ± 0.14 | 98% |
| 8 iterations | 4 | 5.10 ± 0.27 | 0.38 ± 0.15 | 98% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 472.96 ± 0.91 | 417.92 ± 0.64 | 5% |
| 0 iterations | 1 | 444.39 ± 0.82 | 396.41 ± 0.57 | 5% |
| 0 iterations | 2 | 411.74 ± 0.72 | 371.65 ± 0.47 | 5% |
| 0 iterations | 3 | 394.37 ± 0.70 | 356.47 ± 0.45 | 6% |
| 0 iterations | 4 | 383.16 ± 0.68 | 346.50 ± 0.44 | 6% |
| 3 iterations | 0 | 8.13 ± 0.41 | 1.15 ± 0.09 | 98% |
| 3 iterations | 1 | 7.44 ± 0.38 | 0.66 ± 0.07 | 98% |
| 3 iterations | 2 | 7.80 ± 0.40 | 0.54 ± 0.13 | 98% |
| 3 iterations | 3 | 7.54 ± 0.39 | 0.44 ± 0.13 | 99% |
| 3 iterations | 4 | 7.46 ± 0.40 | 0.43 ± 0.13 | 99% |
| 5 iterations | 0 | 6.09 ± 0.27 | 1.48 ± 0.15 | 98% |
| 5 iterations | 1 | 5.91 ± 0.28 | 1.04 ± 0.13 | 98% |
| 5 iterations | 2 | 5.41 ± 0.27 | 0.67 ± 0.12 | 99% |
| 5 iterations | 3 | 5.06 ± 0.25 | 0.51 ± 0.14 | 98% |
| 5 iterations | 4 | 4.88 ± 0.24 | 0.44 ± 0.13 | 98% |
| 8 iterations | 0 | 5.98 ± 0.27 | 1.43 ± 0.14 | 98% |
| 8 iterations | 1 | 5.81 ± 0.27 | 1.07 ± 0.13 | 98% |
| 8 iterations | 2 | 5.35 ± 0.26 | 0.68 ± 0.12 | 99% |
| 8 iterations | 3 | 5.01 ± 0.24 | 0.51 ± 0.13 | 98% |
| 8 iterations | 4 | 4.88 ± 0.25 | 0.44 ± 0.13 | 99% |

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
| starting point L0 | left wing | 2.9 | 4.6 | 9.1 |
| starting point L0 | central | 1.3 | 1.8 | 2.3 |
| starting point L0 | right wing | 1.5 | 4.4 | 75.7 |
| (10,12) 12 iterations | left wing | 3.1 | 4.9 | 10.4 |
| (10,12) 12 iterations | central | 1.4 | 1.8 | 2.6 |
| (10,12) 12 iterations | right wing | 1.5 | 2.9 | 15.4 |
| particle method | left wing | 3.1 | 4.9 | 10.4 |
| particle method | central | 1.4 | 1.9 | 2.6 |
| particle method | right wing | 1.5 | 2.9 | 15.0 |

## Notes

- The chain of (10,12) after 0 iterations ends after 3 of 4 steps: the next step's optimization did not converge.
