# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

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

## Table 2

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
