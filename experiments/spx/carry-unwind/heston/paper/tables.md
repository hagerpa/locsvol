# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

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

## Table 2

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
