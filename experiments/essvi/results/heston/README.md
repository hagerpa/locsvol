# The parametric target with the Heston backbone

Root mean squared error of the implied volatility against the eSSVI surface, in volatility
basis points, over the left wing, the central region and the right wing, and over the central
region of each maturity; point estimates ± two standard errors (paired delete-one jackknife
over the sets of paths).

## On 320 sets of 65,536 independent paths

| Model | left | central | right | 0.25 y | 0.5 y | 1 y | 2 y | 3 y |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 793.8 ± 2.8 | 359.2 ± 0.2 | 532.0 ± 0.8 | 390.2 ± 0.6 | 421.4 ± 0.3 | 390.9 ± 0.3 | 308.6 ± 0.3 | 259.3 ± 0.4 |
| particle method | 8.7 ± 3.7 | 4.6 ± 0.7 | 14.8 ± 0.8 | 7.9 ± 0.8 | 3.9 ± 0.4 | 2.5 ± 0.9 | 3.5 ± 1.5 | 3.3 ± 1.4 |
| (10,12) 5 iterations | 3.3 ± 1.7 | 7.9 ± 1.1 | 11.2 ± 0.9 | 3.4 ± 1.4 | 3.4 ± 1.5 | 6.8 ± 1.5 | 10.1 ± 1.5 | 11.7 ± 1.5 |
| (10,12) 12 iterations | 2.9 ± 2.7 | 2.9 ± 1.0 | 12.2 ± 0.8 | 3.2 ± 1.3 | 1.9 ± 0.9 | 1.9 ± 1.5 | 3.1 ± 1.5 | 3.9 ± 1.2 |
| (6,8) 5 iterations | 3.6 ± 2.4 | 9.2 ± 1.1 | 13.6 ± 0.9 | 9.9 ± 1.3 | 3.5 ± 1.4 | 7.0 ± 1.3 | 10.1 ± 1.5 | 12.7 ± 1.5 |
| (6,8) 12 iterations | 3.3 ± 3.5 | 5.2 ± 0.7 | 15.0 ± 0.9 | 10.0 ± 1.3 | 3.2 ± 0.4 | 1.9 ± 0.7 | 2.4 ± 1.6 | 4.1 ± 1.2 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Model | left | central | right | 0.25 y | 0.5 y | 1 y | 2 y | 3 y |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 1.7 | 0.6 | 7.9 | 0.6 | 0.6 | 0.6 | 0.6 | 0.5 |
| particle method | 2.5 | 0.8 | 0.8 | 0.8 | 0.8 | 0.9 | 0.8 | 0.8 |
| (10,12) 5 iterations | 2.5 | 0.8 | 0.8 | 0.8 | 0.8 | 0.9 | 0.8 | 0.8 |
| (10,12) 12 iterations | 2.5 | 0.8 | 0.8 | 0.8 | 0.8 | 0.9 | 0.8 | 0.8 |
| (6,8) 5 iterations | 2.5 | 0.8 | 0.9 | 0.8 | 0.8 | 0.9 | 0.8 | 0.8 |
| (6,8) 12 iterations | 2.5 | 0.8 | 0.9 | 0.8 | 0.8 | 0.9 | 0.8 | 0.8 |

## On 20 sets of 65,536 Sobol paths

| Model | left | central | right | 0.25 y | 0.5 y | 1 y | 2 y | 3 y |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 792.7 ± 2.9 | 359.3 ± 0.2 | 533.3 ± 2.9 | 389.7 ± 0.4 | 421.6 ± 0.4 | 391.2 ± 0.6 | 308.6 ± 0.7 | 259.8 ± 1.1 |
| (10,12) 3 iterations | 53.3 ± 5.5 | 27.0 ± 1.7 | 24.1 ± 1.8 | 3.4 ± 0.3 | 17.7 ± 1.8 | 36.2 ± 1.7 | 33.8 ± 2.8 | 29.2 ± 2.8 |
| (10,12) 5 iterations | 4.2 ± 4.5 | 6.6 ± 2.2 | 11.2 ± 2.1 | 1.7 ± 0.5 | 1.6 ± 0.9 | 5.4 ± 1.3 | 10.2 ± 3.3 | 8.8 ± 4.2 |
| (10,12) 8 iterations | 2.9 ± 3.3 | 2.1 ± 1.0 | 11.9 ± 2.0 | 1.6 ± 0.4 | 2.1 ± 1.2 | 0.7 ± 0.8 | 3.1 ± 3.5 | 2.0 ± 1.0 |
| (10,12) 12 iterations | 3.1 ± 3.8 | 2.2 ± 1.1 | 12.1 ± 2.0 | 1.6 ± 0.4 | 2.1 ± 1.2 | 0.7 ± 0.6 | 3.2 ± 3.4 | 2.5 ± 0.9 |
| (6,8) 3 iterations | 51.6 ± 5.5 | 27.7 ± 1.6 | 27.3 ± 2.7 | 9.2 ± 0.6 | 17.6 ± 1.8 | 37.0 ± 1.7 | 34.5 ± 2.7 | 29.5 ± 2.7 |
| (6,8) 5 iterations | 3.8 ± 4.1 | 7.7 ± 2.0 | 13.8 ± 3.3 | 8.1 ± 0.6 | 2.3 ± 0.5 | 5.6 ± 1.2 | 10.1 ± 3.4 | 9.7 ± 4.2 |
| (6,8) 8 iterations | 2.3 ± 2.5 | 4.3 ± 0.5 | 14.9 ± 3.1 | 8.2 ± 0.6 | 3.7 ± 1.1 | 1.5 ± 0.9 | 2.4 ± 3.5 | 2.0 ± 0.9 |
| (6,8) 12 iterations | 2.6 ± 3.3 | 4.4 ± 0.5 | 15.0 ± 3.1 | 8.2 ± 0.6 | 3.8 ± 1.1 | 1.7 ± 1.0 | 2.3 ± 3.5 | 2.5 ± 1.1 |
| particle method | 5.0 ± 1.6 | 4.4 ± 0.7 | 14.4 ± 1.8 | 7.3 ± 0.3 | 4.5 ± 1.0 | 2.6 ± 0.6 | 3.6 ± 3.4 | 1.2 ± 1.3 |

Monte Carlo noise alone accounts for the following part of these RMSEs:

| Model | left | central | right | 0.25 y | 0.5 y | 1 y | 2 y | 3 y |
|---|---|---|---|---|---|---|---|---|
| starting point L0 | 2.2 | 0.9 | 33.8 | 0.4 | 0.6 | 0.8 | 1.1 | 1.4 |
| (10,12) 3 iterations | 3.2 | 1.5 | 2.3 | 0.5 | 1.0 | 0.9 | 1.9 | 2.3 |
| (10,12) 5 iterations | 3.1 | 1.4 | 2.5 | 0.4 | 1.0 | 0.9 | 1.8 | 2.2 |
| (10,12) 8 iterations | 3.1 | 1.4 | 2.5 | 0.4 | 1.0 | 0.9 | 1.9 | 2.3 |
| (10,12) 12 iterations | 3.1 | 1.4 | 2.5 | 0.4 | 1.0 | 0.9 | 1.9 | 2.3 |
| (6,8) 3 iterations | 3.2 | 1.5 | 3.0 | 0.4 | 1.0 | 1.0 | 1.9 | 2.3 |
| (6,8) 5 iterations | 3.1 | 1.4 | 2.9 | 0.4 | 1.0 | 0.9 | 1.8 | 2.2 |
| (6,8) 8 iterations | 3.1 | 1.5 | 2.9 | 0.4 | 1.0 | 0.9 | 1.9 | 2.3 |
| (6,8) 12 iterations | 3.1 | 1.5 | 2.9 | 0.4 | 1.0 | 0.9 | 1.9 | 2.3 |
| particle method | 3.1 | 1.4 | 2.5 | 0.4 | 0.9 | 0.9 | 1.8 | 2.2 |

## Smiles

![smiles](figures/smiles.png)

Implied volatilities of the models on the 320 sets of independent paths
at the five maturities, with the eSSVI target.
