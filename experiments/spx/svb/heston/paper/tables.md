# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 8.26 ± 5.38 | 3.03 ± 1.14 | 5.76 ± 0.40 |
| Joint regression (6,8), 5 iterations | 9.64 ± 5.80 | 3.46 ± 0.71 | 14.98 ± 0.40 |
| Particle method | 21.27 ± 7.84 | 10.17 ± 1.17 | 17.09 ± 0.39 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 6.02 | 0.94 | 0.68 |
| Joint regression (6,8), 5 iterations | 6.04 | 0.94 | 0.68 |
| Particle method | 5.77 | 0.93 | 0.65 |

## Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 411.30 ± 2.12 | 312.35 ± 0.70 | 2% |
| (10,12) | 3 iterations | 13.89 ± 1.62 | 4.02 ± 0.27 | 50% |
| (10,12) | 5 iterations | 13.02 ± 1.60 | 4.51 ± 0.32 | 45% |
| (10,12) | 8 iterations | 13.07 ± 1.59 | 4.66 ± 0.31 | 45% |
| (6,8) | 0 iterations | 412.15 ± 2.13 | 312.84 ± 0.70 | 1% |
| (6,8) | 3 iterations | 14.34 ± 1.63 | 5.79 ± 0.16 | 44% |
| (6,8) | 5 iterations | 13.47 ± 1.54 | 6.26 ± 0.26 | 42% |
| (6,8) | 8 iterations | 13.51 ± 1.53 | 6.36 ± 0.26 | 42% |
| Particle method |  | 28.26 ± 0.91 | 23.82 ± 0.78 | 18% |
| Refined SVI target |  | 26.64 | 23.68 | 16% |
