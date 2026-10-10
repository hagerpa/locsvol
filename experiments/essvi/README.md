# The parametric target

This folder calibrates the leverage factor to a parametric implied volatility surface of the
literature: the extended SSVI (eSSVI) surface of Farkas, Ferrari and Ulrych (2022) for SPX on
23 June 2020 at the maturities 0.25, 0.5, 1, 2 and 3 years, the target of the stochastic
Knothe-Rosenblatt calibration of Beiglboeck, Hasenbichler and Pammer (2026), with their reference
Heston and one-factor Bergomi models as backbones. The surface is smooth in time and its Dupire
local volatility is analytic, so the study shows the fixed-point iteration and the particle method
on a target without the seams of a market surface. The results for both backbones are in the
folders `<backbone>/` and summarized in [Results](#results) below.

The study follows the SPX study of the paper ([experiments/spx](../spx)) step by step, with the
same settings (`../spx/settings.py`) wherever they apply; `parameters.py` holds the numbers that
differ, `calibrate.py` runs the study for one backbone and `publish.py` writes its results.

## Target

The eSSVI surface with total variance

    w(T, k) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)),
    phi(theta) = eta theta^-lambda (1 + theta)^(lambda - 1),   rho(theta) = rho_m + (rho_0 - rho_m) exp(-a theta),

with (eta, lambda, rho_m, rho_0, a) = (0.9664, 0.4963, -0.7633, -0.2269, 340.68), theta linear
between the knots given by the ATM volatilities 26.52, 27.17, 25.45, 23.81 and 22.98 % at the five
maturities and continued linearly to the horizon of 3.05 years. The quotes are the surface's
implied volatilities at 41 equally spaced log strikes per maturity on the band |k| < 0.2 + 0.4 T of
the SPX study within [-0.6, 0.4], the range of the smile figure of Beiglboeck, Hasenbichler and
Pammer; they have no bid-ask spread, so there are no correction steps and no bid-ask statistics.
The coordinates of the leverage factor come from the SSVI fit to these quotes as in the SPX study,
and the target local volatility is the eSSVI's own Dupire local volatility, read at x clipped to
the quantiles of the SSVI fit.

## Backbones

The reference models of Beiglboeck, Hasenbichler and Pammer (2026, Table 1), not fitted here:
Heston with (kappa, theta, sigma, rho, v0) = (1.2484, 0.0988, 1.1175, -0.8038, 0.1020), and the
one-factor Bergomi model with kappa = 2.4971, eta = 3.6191, rho = -0.8796 and the piecewise
constant forward variance 0.1021, 0.1196, 0.0908, 0.0846, 0.0838 on the intervals between 0, 0.25,
0.5, 1, 2 and 3 years. In the package's two-factor Bergomi model this is nu = eta / 2, theta = 0
and rho_1 = rho, k_1 = kappa.

## Simulation

The time grid of the SPX study continued to 3.05 years: t_k = (90/365)(k/800)^2 up to 90 days,
then steps of 1/400 with the maturities inserted (1,928 nodes). The calibration paths are 32,768
antithetic paths from scrambled Sobol points with a Brownian bridge, in segments: the points
restart with an independent scramble at 1 and 2 years and the paths of each segment are paired
with those of the preceding ones at random (`locsvol.simulation.brownian_increments`). Over a
single bridge spanning three years the leading Sobol coordinates all go to the coarse shape of the
whole horizon, and the first days, where the local volatility changes fastest, are driven by deep
coordinates; segments of a year give every part of the grid leading coordinates. The 20 evaluation
sets of 65,536 Sobol paths use the same segments; the 320 sets of 65,536 independent paths of the
tables need none.

## Models

The starting point L0 = sigma_Dup, the fixed-point iterates of the bases (10,12) and (6,8) after
3, 5, 8 and 12 iterations, and the particle method, all as in the SPX study.

## Evaluation

As in the SPX study: implied volatilities at 81 equally spaced log strikes per maturity between the
smallest and largest quoted ones, compared with the eSSVI surface over the left wing, the central
region and the right wing, and here also over the central region of each maturity; root mean
squared differences in volatility basis points with jackknife standard errors over the sets of
paths, and the part of each RMSE that Monte Carlo noise alone accounts for.

## Results

Each folder `<backbone>/` holds the results of one run of `calibrate.py`. Left wing, central
region and right wing RMSE against the eSSVI surface, in volatility basis points, on
320 sets of 65,536 independent paths, after 12 fixed-point iterations; after 5, the maturities of two and three
years are still converging, and the reports have those numbers as well:

| Backbone | RMSE, (10,12) 12 iterations | RMSE, (6,8) 12 iterations | RMSE, particle method |
|---|---|---|---|
| [Bergomi](bergomi) | 3.4 / 2.2 / 3.7 | 4.9 / 3.9 / 7.0 | 10.4 / 5.7 / 13.4 |
| [Heston](heston) | 2.9 / 2.9 / 12.2 | 3.3 / 5.2 / 15.0 | 8.7 / 4.6 / 14.8 |

Each folder opens with its report, `README.md`: the tables of the regional and the
per-maturity errors on the independent and on the Sobol paths and the smile figure; beside
it `summary.json` with every model's errors and standard errors and `figures/`.

## Running the study

```sh
python experiments/essvi/calibrate.py --backbone heston --output <folder>
python experiments/essvi/publish.py --backbone heston --output <folder>
```

`calibrate.py` keeps the results of its expensive steps in the output folder and reuses them when
it is run again. `publish.py` writes `<backbone>/`: `README.md` with the tables and the smile
figure, `summary.json` and `figures/`; it also rebuilds the [Results](#results) section above.
