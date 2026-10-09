# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 6.96 ± 4.60 | 6.26 ± 1.00 | 5.81 ± 0.24 |
| Joint regression (6,8), 5 iterations | 13.09 ± 4.45 | 7.68 ± 0.68 | 6.92 ± 0.20 |
| Particle method | 74.21 ± 4.57 | 40.91 ± 1.08 | 21.24 ± 0.22 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 3.53 | 0.90 | 0.52 |
| Joint regression (6,8), 5 iterations | 3.57 | 0.89 | 0.42 |
| Particle method | 3.36 | 0.87 | 0.43 |

## Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 537.74 ± 1.74 | 296.75 ± 0.49 | 2% |
| (10,12) | 3 iterations | 8.55 ± 0.79 | 4.52 ± 0.16 | 49% |
| (10,12) | 5 iterations | 3.37 ± 0.70 | 0.44 ± 0.16 | 91% |
| (10,12) | 8 iterations | 3.45 ± 0.54 | 0.73 ± 0.31 | 80% |
| (6,8) | 0 iterations | 539.03 ± 1.74 | 297.34 ± 0.48 | 2% |
| (6,8) | 3 iterations | 10.67 ± 1.22 | 4.71 ± 0.13 | 51% |
| (6,8) | 5 iterations | 5.59 ± 1.11 | 0.79 ± 0.34 | 86% |
| (6,8) | 8 iterations | 6.28 ± 1.14 | 1.25 ± 0.30 | 76% |
| Particle method |  | 37.13 ± 1.96 | 21.33 ± 0.77 | 41% |
| Refined SVI target |  | 7.37 | 6.40 | 42% |
