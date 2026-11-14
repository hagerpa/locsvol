# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

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

## Table 2

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
