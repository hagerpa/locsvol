# fomc, Bergomi backbone

Interest rate decision of the Federal Open Market Committee; SPX about -3%, VIX from 15 to 28. The Bergomi backbone is fitted to the S&P 500 index options of 2024-11-20 (nu = 2.3192, theta = 0.1685, rho1 = -0.7541, rho2 = -0.2790); the leverage factor is calibrated to the 2,994 quotes of 2024-12-18, 1,503 of them liquid, at 16 expiries from 9 to 394 days.

Contents: [the paper's tables and figures](#the-papers-tables-and-figures), [all models](#all-models), [fixed-point iteration](#fixed-point-iteration), [correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), [prices at every expiry](#prices-at-every-expiry), [Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in [summary.json](summary.json) and the calibrated parameters in [parameters.json](parameters.json).

## The paper's tables and figures

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

### Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 4.58 ± 2.13 | 4.72 ± 0.94 | 3.59 ± 0.44 |
| Joint regression (6,8), 5 iterations | 16.91 ± 4.16 | 4.86 ± 0.73 | 6.20 ± 0.35 |
| Particle method | 46.47 ± 3.97 | 26.17 ± 1.01 | 26.91 ± 0.32 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 3.17 | 0.81 | 0.51 |
| Joint regression (6,8), 5 iterations | 3.19 | 0.81 | 0.43 |
| Particle method | 3.06 | 0.79 | 0.43 |

### Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 433.29 ± 1.50 | 258.78 ± 0.40 | 4% |
| (10,12) | 3 iterations | 7.20 ± 1.15 | 0.52 ± 0.13 | 87% |
| (10,12) | 5 iterations | 3.46 ± 0.74 | 0.30 ± 0.29 | 95% |
| (10,12) | 8 iterations | 3.37 ± 0.70 | 0.29 ± 0.28 | 95% |
| (6,8) | 0 iterations | 434.26 ± 1.50 | 259.16 ± 0.40 | 3% |
| (6,8) | 3 iterations | 11.16 ± 1.32 | 0.85 ± 0.14 | 81% |
| (6,8) | 5 iterations | 5.99 ± 0.97 | 0.81 ± 0.45 | 87% |
| (6,8) | 8 iterations | 6.68 ± 1.05 | 0.96 ± 0.47 | 84% |
| Particle method |  | 24.93 ± 1.43 | 11.18 ± 0.68 | 50% |
| Refined SVI target |  | 7.37 | 6.40 | 42% |

![Smiles](paper/smiles.png)

Implied volatilities at the expiries nearest one week, one month and one year, and their differences to the target ([PDF](paper/smiles.pdf)).

![Convergence](paper/convergence.png)

Root mean square coefficient increment of the fixed-point iteration, and the bid-ask loss along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).

## All models

From here on every model is priced on 20 sets of 65,536 scrambled Sobol paths. Root mean squared error of the implied volatility against the refined SVI target per region and the bid-ask losses, in volatility basis points, ± two standard errors (paired delete-one jackknife over the sets); then the share of liquid quotes inside their bid-ask interval, outside by less than two standard errors of the price (within noise), and further outside.

| Model | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside | Within noise | Further outside |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 914.26 ± 4.54 | 328.98 ± 0.69 | 298.51 ± 2.55 | 572.18 ± 2.93 | 320.15 ± 0.77 | 3% | 0% | 97% |
| (10,12) 3 iterations | 30.44 ± 4.00 | 12.23 ± 0.60 | 15.86 ± 2.76 | 14.14 ± 1.58 | 9.51 ± 0.42 | 38% | 4% | 58% |
| (10,12) 5 iterations | 5.09 ± 2.88 | 5.24 ± 0.86 | 4.29 ± 2.28 | 7.60 ± 0.36 | 5.57 ± 0.33 | 50% | 5% | 45% |
| (10,12) 8 iterations | 4.76 ± 2.49 | 3.17 ± 0.60 | 5.68 ± 3.14 | 7.46 ± 0.36 | 5.76 ± 0.17 | 47% | 6% | 48% |
| (10,12) 12 iterations | 4.78 ± 2.55 | 3.19 ± 0.61 | 5.76 ± 3.17 | 7.45 ± 0.37 | 5.74 ± 0.17 | 47% | 6% | 48% |
| (10,12) 0 iterations + 4 correction steps | 706.87 ± 4.47 | 243.84 ± 0.51 | 302.21 ± 2.57 | 432.25 ± 2.54 | 258.63 ± 0.56 | 3% | 0% | 96% |
| (10,12) 3 iterations + 4 correction steps | 15.28 ± 4.23 | 11.77 ± 0.37 | 13.00 ± 2.39 | 7.01 ± 1.42 | 0.79 ± 0.41 | 88% | 6% | 6% |
| (10,12) 5 iterations + 4 correction steps | 9.07 ± 1.24 | 12.57 ± 0.39 | 9.30 ± 1.32 | 3.54 ± 0.81 | 0.52 ± 0.57 | 93% | 5% | 2% |
| (10,12) 8 iterations + 4 correction steps | 9.08 ± 0.99 | 12.63 ± 0.40 | 9.92 ± 1.90 | 3.49 ± 0.71 | 0.50 ± 0.58 | 93% | 5% | 2% |
| (6,8) 3 iterations | 37.17 ± 3.71 | 12.79 ± 0.47 | 17.91 ± 1.31 | 16.87 ± 1.29 | 10.00 ± 0.30 | 35% | 3% | 63% |
| (6,8) 5 iterations | 17.89 ± 3.61 | 5.66 ± 0.51 | 6.46 ± 1.07 | 10.42 ± 0.77 | 6.38 ± 0.31 | 45% | 4% | 51% |
| (6,8) 8 iterations | 17.12 ± 3.64 | 4.13 ± 0.41 | 7.81 ± 2.02 | 10.24 ± 0.77 | 6.62 ± 0.22 | 42% | 5% | 54% |
| (6,8) 12 iterations | 17.17 ± 3.66 | 4.12 ± 0.42 | 7.95 ± 2.13 | 10.26 ± 0.78 | 6.59 ± 0.22 | 42% | 5% | 53% |
| (6,8) 0 iterations + 4 correction steps | 708.47 ± 4.47 | 244.42 ± 0.51 | 301.17 ± 2.55 | 433.26 ± 2.55 | 259.01 ± 0.56 | 3% | 1% | 96% |
| (6,8) 3 iterations + 4 correction steps | 24.67 ± 3.71 | 11.45 ± 0.23 | 15.67 ± 1.15 | 11.07 ± 1.20 | 1.00 ± 0.19 | 82% | 7% | 11% |
| (6,8) 5 iterations + 4 correction steps | 9.37 ± 3.26 | 12.08 ± 0.26 | 10.66 ± 0.93 | 5.93 ± 0.88 | 1.08 ± 0.44 | 87% | 6% | 7% |
| (6,8) 8 iterations + 4 correction steps | 13.89 ± 3.67 | 11.90 ± 0.28 | 10.66 ± 1.51 | 6.81 ± 0.99 | 1.27 ± 0.51 | 85% | 7% | 8% |
| particle method | 44.71 ± 2.65 | 25.99 ± 0.27 | 26.83 ± 0.48 | 24.37 ± 0.76 | 11.12 ± 0.26 | 50% | 4% | 46% |
| refined SVI target |  |  |  | 7.37 | 6.40 | 42% |  |  |

## Fixed-point iteration

![Iterations](figures/iterations.png)

The root mean square coefficient increment of every iteration, and every metric of the iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method as a horizontal line. Change of every iterate against the iterate after 12 iterations, ± two paired standard errors:

| Iterate | Left wing | Central | Right wing | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes |
|---|---|---|---|---|---|
| (10,12) 3 iterations | 25.66 ± 2.29 | 9.04 ± 1.11 | 10.10 ± 5.90 | 6.69 ± 1.31 | 3.78 ± 0.53 |
| (10,12) 5 iterations | 0.31 ± 0.51 | 2.05 ± 0.29 | -1.47 ± 0.94 | 0.15 ± 0.04 | -0.17 ± 0.22 |
| (10,12) 8 iterations | -0.01 ± 0.09 | -0.02 ± 0.01 | -0.09 ± 0.03 | 0.01 ± 0.01 | 0.02 ± 0.00 |
| (6,8) 3 iterations | 20.00 ± 1.69 | 8.67 ± 0.79 | 9.97 ± 3.24 | 6.61 ± 0.69 | 3.41 ± 0.47 |
| (6,8) 5 iterations | 0.72 ± 0.28 | 1.55 ± 0.21 | -1.49 ± 1.15 | 0.16 ± 0.04 | -0.21 ± 0.12 |
| (6,8) 8 iterations | -0.05 ± 0.05 | 0.01 ± 0.02 | -0.13 ± 0.12 | -0.02 ± 0.01 | 0.03 ± 0.00 |

## Correction steps

![Corrections](figures/corrections.png)

Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, ± two standard errors.

### Basis (10,12)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 572.18 ± 2.93 | 320.15 ± 0.77 | 3% |
| 0 iterations | 1 | 518.13 ± 2.74 | 298.85 ± 0.70 | 4% |
| 0 iterations | 2 | 466.17 ± 2.62 | 275.48 ± 0.63 | 3% |
| 0 iterations | 3 | 443.91 ± 2.58 | 264.82 ± 0.58 | 3% |
| 0 iterations | 4 | 432.25 ± 2.54 | 258.63 ± 0.56 | 3% |
| 3 iterations | 0 | 14.14 ± 1.58 | 9.51 ± 0.42 | 38% |
| 3 iterations | 1 | 10.34 ± 1.37 | 6.25 ± 0.23 | 48% |
| 3 iterations | 2 | 7.74 ± 1.22 | 3.23 ± 0.23 | 62% |
| 3 iterations | 3 | 7.16 ± 1.33 | 1.57 ± 0.32 | 76% |
| 3 iterations | 4 | 7.01 ± 1.42 | 0.79 ± 0.41 | 88% |
| 5 iterations | 0 | 7.60 ± 0.36 | 5.57 ± 0.33 | 50% |
| 5 iterations | 1 | 6.60 ± 0.47 | 3.34 ± 0.46 | 57% |
| 5 iterations | 2 | 5.19 ± 0.73 | 1.52 ± 0.66 | 74% |
| 5 iterations | 3 | 3.85 ± 0.81 | 0.66 ± 0.62 | 91% |
| 5 iterations | 4 | 3.54 ± 0.81 | 0.52 ± 0.57 | 93% |
| 8 iterations | 0 | 7.46 ± 0.36 | 5.76 ± 0.17 | 47% |
| 8 iterations | 1 | 6.61 ± 0.40 | 3.60 ± 0.33 | 54% |
| 8 iterations | 2 | 5.24 ± 0.59 | 1.68 ± 0.61 | 70% |
| 8 iterations | 3 | 4.01 ± 0.70 | 0.75 ± 0.67 | 89% |
| 8 iterations | 4 | 3.49 ± 0.71 | 0.50 ± 0.58 | 93% |

### Basis (6,8)

| Started after | Steps | Bid-ask loss, all quotes | Bid-ask loss, liquid quotes | Liquid inside |
|---|---|---|---|---|
| 0 iterations | 0 | 572.18 ± 2.93 | 320.15 ± 0.77 | 3% |
| 0 iterations | 1 | 518.59 ± 2.75 | 298.83 ± 0.71 | 4% |
| 0 iterations | 2 | 467.07 ± 2.62 | 275.75 ± 0.63 | 3% |
| 0 iterations | 3 | 444.58 ± 2.58 | 265.22 ± 0.59 | 3% |
| 0 iterations | 4 | 433.26 ± 2.55 | 259.01 ± 0.56 | 3% |
| 3 iterations | 0 | 16.87 ± 1.29 | 10.00 ± 0.30 | 35% |
| 3 iterations | 1 | 13.00 ± 1.10 | 6.62 ± 0.21 | 45% |
| 3 iterations | 2 | 11.13 ± 1.08 | 3.57 ± 0.16 | 59% |
| 3 iterations | 3 | 10.95 ± 1.14 | 1.87 ± 0.19 | 71% |
| 3 iterations | 4 | 11.07 ± 1.20 | 1.00 ± 0.19 | 82% |
| 5 iterations | 0 | 10.42 ± 0.77 | 6.38 ± 0.31 | 45% |
| 5 iterations | 1 | 9.30 ± 0.81 | 3.97 ± 0.41 | 56% |
| 5 iterations | 2 | 7.61 ± 0.84 | 2.33 ± 0.47 | 69% |
| 5 iterations | 3 | 6.50 ± 0.85 | 1.56 ± 0.48 | 81% |
| 5 iterations | 4 | 5.93 ± 0.88 | 1.08 ± 0.44 | 87% |
| 8 iterations | 0 | 10.24 ± 0.77 | 6.62 ± 0.22 | 42% |
| 8 iterations | 1 | 9.27 ± 0.74 | 4.34 ± 0.37 | 53% |
| 8 iterations | 2 | 7.67 ± 0.74 | 2.72 ± 0.52 | 65% |
| 8 iterations | 3 | 6.55 ± 0.72 | 1.90 ± 0.56 | 79% |
| 8 iterations | 4 | 6.81 ± 0.99 | 1.27 ± 0.51 | 85% |

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
| starting point L0 | left wing | 5.1 | 8.2 | 12.3 |
| starting point L0 | central | 1.2 | 2.4 | 4.4 |
| starting point L0 | right wing | 3.3 | 10.9 | 21.6 |
| (10,12) 12 iterations | left wing | 4.7 | 10.4 | 21.9 |
| (10,12) 12 iterations | central | 1.1 | 2.8 | 5.3 |
| (10,12) 12 iterations | right wing | 2.2 | 7.6 | 18.2 |
| particle method | left wing | 4.2 | 7.9 | 12.7 |
| particle method | central | 1.0 | 1.7 | 3.0 |
| particle method | right wing | 1.8 | 4.0 | 6.1 |
