# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

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

## Table 2

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
