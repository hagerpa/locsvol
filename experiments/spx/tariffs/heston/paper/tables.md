# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 7.08 ± 4.62 | 5.88 ± 1.01 | 14.89 ± 0.62 |
| Joint regression (6,8), 5 iterations | 8.44 ± 4.73 | 5.87 ± 0.95 | 17.65 ± 0.61 |
| Particle method | 45.99 ± 3.88 | 24.28 ± 0.81 | 26.43 ± 0.40 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 3.05 | 0.98 | 0.54 |
| Joint regression (6,8), 5 iterations | 3.05 | 0.98 | 0.54 |
| Particle method | 2.98 | 0.97 | 0.54 |

## Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 497.23 ± 1.10 | 448.27 ± 0.63 | 3% |
| (10,12) | 3 iterations | 9.70 ± 0.95 | 0.18 ± 0.04 | 100% |
| (10,12) | 5 iterations | 6.75 ± 0.84 | 0.23 ± 0.10 | 99% |
| (10,12) | 8 iterations | 6.70 ± 0.83 | 0.29 ± 0.08 | 99% |
| (6,8) | 0 iterations | 574.16 ± 1.31 | 503.09 ± 0.86 | 4% |
| (6,8) | 3 iterations | 10.05 ± 0.96 | 0.15 ± 0.03 | 100% |
| (6,8) | 5 iterations | 6.80 ± 0.85 | 0.29 ± 0.21 | 99% |
| (6,8) | 8 iterations | 6.65 ± 0.82 | 0.39 ± 0.18 | 99% |
| Refined SVI target |  | 6.37 | 1.13 | 98% |
