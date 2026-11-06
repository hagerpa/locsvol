# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

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

## Table 2

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
| Refined SVI target |  | 7.37 | 6.40 | 42% |
