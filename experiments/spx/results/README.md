# Results of the SPX study

Each folder `<pair>/<backbone>/` holds the results of one run of `../run.py`: the backbone (Bergomi or Heston) is fitted to the S&P 500 index options of the earlier day and the leverage factor to those of the event day. Every run is evaluated on 20 sets of 65,536 scrambled Sobol paths.

| Date pair | Backbone | RMSE, (10,12) after 5 iterations | RMSE, particle method | Liquid inside after correction | Liquid inside, particle method |
|---|---|---|---|---|---|
| [`carry-unwind`](carry-unwind/bergomi) (2024-07-08 → 2024-08-05) | [Bergomi](carry-unwind/bergomi) | 7.0 / 4.0 / 5.3 | 31.1 / 23.4 / 12.2 | 98% | 67% |
| [`election`](election/bergomi) (2024-10-09 → 2024-11-06) | [Bergomi](election/bergomi) | 9.5 / 5.6 / 6.5 | 26.0 / 13.4 / 19.8 | 70% | 30% |
| [`election`](election/heston) (2024-10-09 → 2024-11-06) | [Heston](election/heston) | 9.7 / 8.3 / 14.4 | 38.3 / 20.0 / 20.2 | 74% | 30% |
| [`fomc`](fomc/bergomi) (2024-11-20 → 2024-12-18) | [Bergomi](fomc/bergomi) | 5.1 / 5.2 / 4.3 | 44.7 / 26.0 / 26.8 | 93% | 50% |
| [`fomc`](fomc/heston) (2024-11-20 → 2024-12-18) | [Heston](fomc/heston) | 6.5 / 5.5 / 5.7 | 71.3 / 40.0 / 21.4 | 91% | 40% |
| [`tariffs`](tariffs/bergomi) (2025-02-19 → 2025-04-04) | [Bergomi](tariffs/bergomi) | 3.2 / 3.8 / 4.8 | 31.8 / 17.5 / 32.6 | 98% | 83% |
| [`tariffs`](tariffs/heston) (2025-02-19 → 2025-04-04) | [Heston](tariffs/heston) | 4.0 / 6.7 / 14.3 | 50.1 / 25.0 / 25.3 | 99% | 75% |

The RMSE columns give the root mean squared difference between model and target implied volatilities over the left wing, the central region and the right wing, in volatility basis points. The last two columns give the share of liquid quotes inside their bid-ask interval after four correction steps started after 5 iterations, and for the particle method.

- `carry-unwind`: unwind of the yen carry trade; VIX intraday above 60, SPX about -3%
- `election`: day after the US presidential election; event premium released, SPX about +2.5%
- `fomc`: interest rate decision of the Federal Open Market Committee; SPX about -3%, VIX from 15 to 28
- `tariffs`: second day after the tariff announcement of 2 April 2025; SPX about -6%, VIX 45; the earlier day is six weeks before, since March 2025 was itself agitated

## Files

Each folder opens with its report, `README.md`: the paper's tables and figures for that date pair and backbone, every model's errors and bid-ask losses, the fixed-point iteration iterate by iterate, every correction chain, and the smiles and prices at every expiry. Besides it:

- `paper/`: the paper's tables (`tables.md`, `tables.json`) and figures (PDF and PNG), from 320 sets of 65,536 independent paths.
- `summary.json`: for every model (the starting point L0, the fixed-point iterates after 3, 5, 8 and 12 iterations of both bases, every correction step and the particle method) the RMSE per region and the bid-ask loss over all and over the liquid quotes, each with its standard error, and the shares of liquid quotes inside, within Monte Carlo noise of and clearly outside their intervals; also the bid-ask statistics of the target surface.
- `parameters.json`: the fitted backbone, the SSVI and refined SVI target, the coordinates and the coefficients of every parametrized leverage factor, with the code commit and the SHA-256 of both data files.
- `figures/`: the figures of the report.

The quotes are OptionMetrics data from Wharton Research Data Services (WRDS) and are not part of the repository; see `../README.md` for obtaining them and running the study.
