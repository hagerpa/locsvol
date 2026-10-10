# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 11.54 ± 9.20 | 3.77 ± 1.09 | 7.12 ± 0.66 |
| Joint regression (6,8), 5 iterations | 10.15 ± 8.53 | 3.10 ± 0.73 | 10.43 ± 0.49 |
| Particle method | 32.84 ± 8.85 | 11.09 ± 1.17 | 15.51 ± 0.51 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 5.49 | 0.91 | 0.69 |
| Joint regression (6,8), 5 iterations | 5.52 | 0.91 | 0.68 |
| Particle method | 5.28 | 0.90 | 0.66 |

## Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 400.08 ± 2.33 | 295.50 ± 0.67 | 2% |
| (10,12) | 3 iterations | 14.78 ± 1.15 | 3.41 ± 0.71 | 53% |
| (10,12) | 5 iterations | 12.31 ± 0.73 | 4.71 ± 0.89 | 42% |
| (10,12) | 8 iterations | 12.22 ± 0.73 | 4.70 ± 0.89 | 42% |
| (6,8) | 0 iterations | 401.17 ± 2.34 | 296.02 ± 0.67 | 2% |
| (6,8) | 3 iterations | 15.34 ± 1.04 | 4.90 ± 0.61 | 51% |
| (6,8) | 5 iterations | 12.87 ± 0.69 | 5.93 ± 0.77 | 44% |
| (6,8) | 8 iterations | 12.73 ± 0.68 | 5.91 ± 0.77 | 44% |
| Particle method |  | 30.02 ± 1.41 | 23.80 ± 0.78 | 16% |
| Refined SVI target |  | 26.64 | 23.68 | 16% |
