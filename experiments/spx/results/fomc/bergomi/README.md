# fomc, Bergomi backbone

Interest rate decision of the Federal Open Market Committee; SPX about -3%, VIX from 15 to 28. The Bergomi backbone is fitted to the S&P 500 index options of 2024-11-20 (nu = 2.3192, theta = 0.1685, rho1 = -0.7541, rho2 = -0.2790); the leverage factor is calibrated to the 2,994 quotes of 2024-12-18, 1,503 of them liquid, at 16 expiries from 9 to 394 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 5.25 ± 2.73 | 4.62 ± 0.93 | 3.53 ± 0.49 |
| Joint regression (6,8), 5 iterations | 19.58 ± 4.21 | 4.80 ± 0.73 | 6.13 ± 0.35 |
| Particle method | 46.90 ± 3.98 | 26.20 ± 1.02 | 26.93 ± 0.32 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 3.22 | 0.83 | 0.66 |
| Joint regression (6,8), 5 iterations | 3.20 | 0.81 | 0.42 |
| Particle method | 3.05 | 0.79 | 0.43 |

### Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 521.52 ± 1.60 | 299.35 ± 0.50 | 4% |
| (10,12) | 3 iterations | 7.45 ± 1.23 | 0.48 ± 0.06 | 86% |
| (10,12) | 5 iterations | 3.52 ± 0.71 | 0.46 ± 0.39 | 94% |
| (10,12) | 8 iterations | 3.48 ± 0.64 | 0.46 ± 0.39 | 94% |
| (6,8) | 0 iterations | 437.25 ± 1.50 | 259.80 ± 0.40 | 3% |
| (6,8) | 3 iterations | 12.52 ± 1.37 | 0.85 ± 0.11 | 81% |
| (6,8) | 5 iterations | 6.64 ± 1.07 | 0.67 ± 0.42 | 87% |
| (6,8) | 8 iterations | 6.71 ± 1.02 | 0.98 ± 0.48 | 84% |
| Particle method |  | 25.04 ± 1.44 | 11.22 ± 0.69 | 50% |
| Refined SVI target |  | 7.37 | 6.40 | 42% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square coefficient increment of the fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 921.40 ± 4.44 | 330.05 ± 0.68 | 298.92 ± 2.27 | 576.82 ± 2.87 | 321.27 ± 0.76 | 3% | 0% | 97% |
| (10,12) 3 iterations | 30.16 ± 4.53 | 12.44 ± 0.70 | 15.96 ± 3.77 | 14.09 ± 1.85 | 9.65 ± 0.51 | 38% | 4% | 58% |
| (10,12) 5 iterations | 5.99 ± 3.78 | 5.15 ± 1.14 | 4.37 ± 2.87 | 7.78 ± 0.44 | 5.55 ± 0.41 | 50% | 7% | 43% |
| (10,12) 8 iterations | 5.57 ± 3.34 | 3.06 ± 0.66 | 5.77 ± 3.94 | 7.64 ± 0.44 | 5.73 ± 0.17 | 47% | 6% | 47% |
| (10,12) 12 iterations | 5.60 ± 3.39 | 3.07 ± 0.65 | 5.82 ± 3.90 | 7.64 ± 0.45 | 5.71 ± 0.17 | 47% | 6% | 47% |
| (10,12) 0 iterations + 1 correction step | 843.29 ± 4.34 | 295.30 ± 0.62 | 304.56 ± 2.28 | 520.89 ± 2.70 | 299.34 ± 0.70 | 4% | 0% | 96% |
| (10,12) 3 iterations + 4 correction steps | 16.61 ± 4.84 | 11.47 ± 0.36 | 13.24 ± 3.28 | 7.43 ± 1.69 | 0.67 ± 0.36 | 87% | 8% | 5% |
| (10,12) 5 iterations + 4 correction steps | 8.66 ± 1.24 | 12.94 ± 0.49 | 9.37 ± 1.56 | 3.64 ± 0.79 | 0.75 ± 0.77 | 93% | 6% | 2% |
| (10,12) 8 iterations + 4 correction steps | 8.62 ± 1.01 | 12.89 ± 0.47 | 10.11 ± 2.22 | 3.64 ± 0.63 | 0.74 ± 0.75 | 93% | 6% | 2% |
| (6,8) 3 iterations | 39.76 ± 3.70 | 12.59 ± 0.47 | 17.71 ± 1.28 | 17.79 ± 1.29 | 9.94 ± 0.29 | 34% | 3% | 63% |
| (6,8) 5 iterations | 20.22 ± 3.66 | 5.57 ± 0.48 | 6.34 ± 0.95 | 11.10 ± 0.85 | 6.32 ± 0.29 | 45% | 4% | 51% |
| (6,8) 8 iterations | 19.37 ± 3.79 | 4.16 ± 0.41 | 7.68 ± 1.98 | 10.92 ± 0.86 | 6.60 ± 0.21 | 41% | 5% | 54% |
| (6,8) 12 iterations | 19.40 ± 3.85 | 4.15 ± 0.44 | 7.88 ± 2.17 | 10.94 ± 0.87 | 6.57 ± 0.21 | 42% | 5% | 54% |
| (6,8) 0 iterations + 4 correction steps | 713.61 ± 4.35 | 245.10 ± 0.50 | 301.60 ± 2.39 | 436.23 ± 2.50 | 259.70 ± 0.55 | 3% | 0% | 96% |
| (6,8) 3 iterations + 4 correction steps | 28.35 ± 3.69 | 11.33 ± 0.23 | 15.47 ± 1.14 | 12.39 ± 1.23 | 1.00 ± 0.19 | 81% | 7% | 12% |
| (6,8) 5 iterations + 4 correction steps | 12.04 ± 2.89 | 12.08 ± 0.25 | 10.09 ± 0.83 | 6.49 ± 0.93 | 0.92 ± 0.41 | 87% | 6% | 6% |
| (6,8) 8 iterations + 4 correction steps | 11.37 ± 3.47 | 11.85 ± 0.30 | 10.53 ± 1.48 | 6.60 ± 0.88 | 1.28 ± 0.51 | 84% | 8% | 8% |
| particle method | 45.08 ± 2.66 | 25.99 ± 0.27 | 26.78 ± 0.45 | 24.46 ± 0.77 | 11.15 ± 0.26 | 50% | 4% | 46% |
| refined SVI target |  |  |  | 7.37 | 6.40 | 42% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square coefficient increment of every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 24.56 ± 2.23 | 9.37 ± 1.27 | 10.14 ± 7.60 | 6.45 ± 1.51 | 3.94 ± 0.62 |
| (10,12) 5 iterations | 0.39 ± 0.55 | 2.08 ± 0.52 | -1.45 ± 1.08 | 0.14 ± 0.04 | -0.16 ± 0.32 |
| (10,12) 8 iterations | -0.03 ± 0.09 | -0.01 ± 0.01 | -0.05 ± 0.06 | 0.01 ± 0.01 | 0.02 ± 0.00 |
| (6,8) 3 iterations | 20.36 ± 1.74 | 8.44 ± 0.77 | 9.83 ± 3.22 | 6.85 ± 0.63 | 3.36 ± 0.45 |
| (6,8) 5 iterations | 0.81 ± 0.74 | 1.43 ± 0.26 | -1.54 ± 1.39 | 0.15 ± 0.13 | -0.26 ± 0.11 |
| (6,8) 8 iterations | -0.04 ± 0.13 | 0.01 ± 0.04 | -0.20 ± 0.24 | -0.02 ± 0.02 | 0.02 ± 0.00 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 576.82 ± 2.87 | 321.27 ± 0.76 | 3% |
| 0 iterations | 1 | 520.89 ± 2.70 | 299.34 ± 0.70 | 4% |
| 3 iterations | 0 | 14.09 ± 1.85 | 9.65 ± 0.51 | 38% |
| 3 iterations | 1 | 10.33 ± 1.78 | 6.35 ± 0.29 | 48% |
| 3 iterations | 2 | 7.95 ± 1.59 | 3.22 ± 0.23 | 62% |
| 3 iterations | 3 | 7.59 ± 1.61 | 1.46 ± 0.26 | 76% |
| 3 iterations | 4 | 7.43 ± 1.69 | 0.67 ± 0.36 | 87% |
| 5 iterations | 0 | 7.78 ± 0.44 | 5.55 ± 0.41 | 50% |
| 5 iterations | 1 | 6.87 ± 0.56 | 3.35 ± 0.63 | 56% |
| 5 iterations | 2 | 5.13 ± 0.68 | 1.69 ± 0.86 | 75% |
| 5 iterations | 3 | 3.90 ± 0.77 | 0.87 ± 0.79 | 90% |
| 5 iterations | 4 | 3.64 ± 0.79 | 0.75 ± 0.77 | 93% |
| 8 iterations | 0 | 7.64 ± 0.44 | 5.73 ± 0.17 | 47% |
| 8 iterations | 1 | 6.87 ± 0.48 | 3.57 ± 0.38 | 54% |
| 8 iterations | 2 | 5.33 ± 0.56 | 1.83 ± 0.75 | 71% |
| 8 iterations | 3 | 4.05 ± 0.64 | 0.93 ± 0.77 | 89% |
| 8 iterations | 4 | 3.64 ± 0.63 | 0.74 ± 0.75 | 93% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 576.82 ± 2.87 | 321.27 ± 0.76 | 3% |
| 0 iterations | 1 | 521.63 ± 2.69 | 299.54 ± 0.70 | 4% |
| 0 iterations | 2 | 470.29 ± 2.58 | 276.54 ± 0.62 | 3% |
| 0 iterations | 3 | 447.34 ± 2.53 | 265.86 ± 0.58 | 3% |
| 0 iterations | 4 | 436.23 ± 2.50 | 259.70 ± 0.55 | 3% |
| 3 iterations | 0 | 17.79 ± 1.29 | 9.94 ± 0.29 | 34% |
| 3 iterations | 1 | 13.94 ± 1.11 | 6.58 ± 0.20 | 45% |
| 3 iterations | 2 | 12.21 ± 1.11 | 3.58 ± 0.15 | 59% |
| 3 iterations | 3 | 12.19 ± 1.18 | 1.87 ± 0.18 | 70% |
| 3 iterations | 4 | 12.39 ± 1.23 | 1.00 ± 0.19 | 81% |
| 5 iterations | 0 | 11.10 ± 0.85 | 6.32 ± 0.29 | 45% |
| 5 iterations | 1 | 10.05 ± 0.88 | 3.89 ± 0.38 | 55% |
| 5 iterations | 2 | 8.44 ± 0.93 | 2.21 ± 0.44 | 69% |
| 5 iterations | 3 | 6.89 ± 0.87 | 1.42 ± 0.45 | 82% |
| 5 iterations | 4 | 6.49 ± 0.93 | 0.92 ± 0.41 | 87% |
| 8 iterations | 0 | 10.92 ± 0.86 | 6.60 ± 0.21 | 41% |
| 8 iterations | 1 | 10.00 ± 0.83 | 4.30 ± 0.36 | 53% |
| 8 iterations | 2 | 8.72 ± 0.89 | 2.63 ± 0.51 | 65% |
| 8 iterations | 3 | 7.08 ± 0.81 | 1.80 ± 0.55 | 79% |
| 8 iterations | 4 | 6.60 ± 0.88 | 1.28 ± 0.51 | 84% |

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
| starting point L0 | left wing | 5.0 | 8.0 | 12.2 |
| starting point L0 | central | 1.2 | 2.3 | 3.7 |
| starting point L0 | right wing | 3.2 | 9.9 | 16.7 |
| (10,12) 12 iterations | left wing | 4.9 | 11.5 | 22.0 |
| (10,12) 12 iterations | central | 1.1 | 3.4 | 5.7 |
| (10,12) 12 iterations | right wing | 2.2 | 9.2 | 25.5 |
| particle method | left wing | 4.1 | 7.7 | 12.7 |
| particle method | central | 0.9 | 1.7 | 3.0 |
| particle method | right wing | 1.7 | 3.8 | 6.1 |

## Notes

- The chain of (10,12) after 0 iterations ends after 1 of 4 steps: the next step's optimization did not converge.
