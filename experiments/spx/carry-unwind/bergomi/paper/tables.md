# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

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

## Table 2

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
| Refined SVI target |  | 9.89 | 8.06 | 77% |
