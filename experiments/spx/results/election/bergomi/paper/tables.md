# Tables of the paper

Both tables are computed on 320 replications of 65,536 independent pseudo-random paths. Entries are point estimates ± two standard errors (paired delete-one jackknife over the replications), in volatility basis points.

## Table 1

Root mean squared error (RMSE) of the implied volatility against the refined SVI target after 5 fixed-point iterations, with the particle method as a reference.

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 5.80 ± 3.97 | 5.95 ± 0.85 | 7.50 ± 0.51 |
| Joint regression (6,8), 5 iterations | 13.03 ± 5.58 | 6.06 ± 0.82 | 8.39 ± 0.46 |
| Particle method | 28.91 ± 6.33 | 14.31 ± 1.15 | 19.47 ± 0.46 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Method | Left | Central | Right |
|---|---|---|---|
| Joint regression (10,12), 5 iterations | 5.03 | 0.95 | 0.54 |
| Joint regression (6,8), 5 iterations | 5.10 | 0.96 | 0.54 |
| Particle method | 4.76 | 0.92 | 0.56 |

## Table 2

Bid-ask loss and share of liquid quotes inside their bid-ask interval after the correction steps.

| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |
|---|---|---|---|---|
| (10,12) | 0 iterations | 613.33 ± 2.33 | 280.96 ± 0.34 | 1% |
| (10,12) | 3 iterations | 14.62 ± 0.90 | 7.32 ± 0.45 | 28% |
| (10,12) | 5 iterations | 6.45 ± 0.69 | 1.78 ± 0.39 | 70% |
| (10,12) | 8 iterations | 6.39 ± 0.54 | 2.07 ± 0.39 | 65% |
| (6,8) | 0 iterations | 616.33 ± 2.34 | 281.85 ± 0.34 | 1% |
| (6,8) | 3 iterations | 18.22 ± 1.19 | 7.49 ± 0.43 | 29% |
| (6,8) | 5 iterations | 9.88 ± 1.20 | 2.52 ± 0.34 | 61% |
| (6,8) | 8 iterations | 9.55 ± 1.10 | 2.90 ± 0.35 | 55% |
| Particle method |  | 18.76 ± 1.73 | 7.40 ± 0.22 | 31% |
| Refined SVI target |  | 11.00 | 10.98 | 22% |
