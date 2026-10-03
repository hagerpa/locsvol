# Results of the SPX study

Each folder `<pair>/<backbone>/` holds the results of one run of `../run.py`: the backbone (Bergomi or Heston) is fitted to the S&P 500 index options of the earlier day and the leverage factor to those of the event day. Every run is evaluated on 20 sets of 65,536 scrambled Sobol paths.

| Date pair | Backbone | RMSE, (10,12) after 5 iterations | RMSE, particle method | Liquid inside after correction | Liquid inside, particle method |
|---|---|---|---|---|---|
| `fomc` (2024-11-20 → 2024-12-18) | Bergomi | 6.0 / 5.1 / 4.4 | 45.1 / 26.0 / 26.8 | 93% | 50% |

The RMSE columns give the root mean squared difference between model and target implied volatilities over the left wing, the central region and the right wing, in volatility basis points. The last two columns give the share of liquid quotes inside their bid-ask interval after four correction steps started after 5 iterations, and for the particle method.

- `fomc`: interest rate decision of the Federal Open Market Committee; SPX about -3%, VIX from 15 to 28

## Files

- `summary.json`: for every model (the starting point L0, the fixed-point iterates after 3, 5, 8 and 12 iterations of both bases, every correction step and the particle method) the RMSE per region and the bid-ask loss over all and over the liquid quotes, each with its standard error, and the shares of liquid quotes inside, within Monte Carlo noise of and clearly outside their intervals; also the bid-ask statistics of the target surface.
- `parameters.json`: the fitted backbone, the SSVI and refined SVI target, the coordinates and the coefficients of every parametrized leverage factor, with the code commit and the SHA-256 of both data files.
- `smiles.png`, `convergence.png`, `spreads.png`: the figures of the run, drawn as in the paper.
- `fomc/bergomi/paper/`: the paper's tables and figures, from 320 sets of 65,536 independent paths.

The quotes are OptionMetrics data from Wharton Research Data Services (WRDS) and are not part of the repository; see `../README.md` for obtaining them and running the study.
