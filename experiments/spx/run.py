"""The study of Section 3 of the paper for one date pair and backbone.

    python experiments/spx/run.py --pair fomc --backbone bergomi --output <folder>

1. The quotes of the earlier day, their target and the fit of the backbone (Section 3.2).
2. The quotes of the event day and their target: SSVI, refined SVI, Dupire (Section 3.1).
3. The time grid, the coordinates, the base L0 = min(sigma_Dup, 5) and the target, tabulated; the
   training paths and the envelope of the coordinates (Section 3.3).
4. Twelve fixed-point iterations for each basis (Algorithm 2.1).
5. The particle method of Guyon and Henry-Labordere on the same paths.
6. Four correction steps after 0, 3, 5 and 8 iterations for each basis.
7. Prices of all models on 20 sets of 65,536 Sobol paths, and of the models of the paper's tables
   and figures on 320 sets of 65,536 independent paths (Section 3.4).

Every step keeps its result in the output folder and reuses it when the script runs again.
results.py turns the prices into the results of results/<pair>/<backbone>/.
"""

import argparse
import time
from functools import cache
from hashlib import sha256
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import settings as s
from date_pairs import PAIRS
from evaluation import Comparison
from quotes import block_weights, read_quotes

import locsvol as lsv
from locsvol.backbones import Bergomi, ForwardVariance
from locsvol.calibration import (
    correction_steps,
    fit_bergomi,
    fit_heston,
    fixed_point,
    particle_method,
    support_points,
    variance_swap_volatility,
    widened_intervals,
)
from locsvol.leverage import Capped, Chart, Parametrized, envelope, table_times, tabulate
from locsvol.pricing import monte_carlo_prices, replicated_prices
from locsvol.regression import ChebyshevTensor, GammaRegression
from locsvol.simulation import (
    brownian_increments,
    graded_times,
    simulate,
    trapezoid_weights,
    with_dates,
)
from locsvol.targets import Dupire, digital_quantiles, fit_ssvi, fit_svi, forward_variance


def kept(path, compute):
    """The result saved at path; computed and saved there first if it is not."""
    if not path.exists():
        started = time.time()
        lsv.save(path, compute())
        print(f"{path.name}: {time.time() - started:.0f} s", flush=True)
    return lsv.load(path)


def day_quotes(data, day, expected_sha256):
    """The selected quotes of a day, from the file whose SHA-256 date_pairs.py lists."""
    path = data / f"spx_optionmetrics_{day.isoformat()}.csv"
    if sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"{path.name} differs from the file of date_pairs.py")
    return read_quotes(path, day)


def time_grid(quotes):
    """The graded grid of the paper with the quoted maturities as nodes."""
    times = graded_times(s.HORIZON, s.GRADED_HORIZON, s.GRADED_STEPS, s.GRADING_POWER, s.TAIL_STEP)
    return with_dates(times, quotes.maturities)


def fit_target(quotes):
    """The SSVI surface and the refined SVI surface of the quotes."""
    data = (quotes.maturities, quotes.log_strikes, quotes.implied_volatility, quotes.weights)
    ssvi = fit_ssvi(*data)
    return dict(ssvi=ssvi, surface=fit_svi(*data, ssvi))


def fit_backbone(backbone, quotes, surface, streams):
    """The backbone fitted to the mid prices of the earlier day beyond 0.05 years; for Bergomi
    with the forward variance of that day, then set to 1 for the event day."""
    maturities = np.asarray(quotes.maturities)
    weights = block_weights(
        maturities.tolist(), np.asarray(quotes.is_put).tolist(), maturities > s.BACKBONE_FIT_FROM
    )
    if backbone == "heston":
        return fit_heston(
            quotes.maturities,
            quotes.log_strikes,
            quotes.mid,
            weights,
            quotes.vega,
            floor=s.HESTON_FLOOR,
        )
    times = time_grid(quotes)
    xi = ForwardVariance(times[1:], forward_variance(surface, times))
    template = Bergomi(**s.BERGOMI_START, **s.BERGOMI_FIXED, forward_variance=xi)
    increments = [
        brownian_increments(times, streams["fit"](r), s.TRAINING_PATHS, 3)
        for r in range(s.FIT_REPLICATIONS)
    ]
    calls = jnp.ones(quotes.maturities.size, dtype=bool)
    steps = fit_bergomi(
        template,
        times,
        increments,
        quotes.maturities,
        quotes.log_strikes,
        calls,
        quotes.mid,
        quotes.mid,
        weights,
        quotes.vega,
        s.BERGOMI_RADII,
        s.BERGOMI_CUMULATIVE_RADIUS,
    )
    fitted = steps[-1]
    return Bergomi(fitted.nu, fitted.theta, fitted.rho1, fitted.rho2, **s.BERGOMI_FIXED)


def leverage_inputs(target, times):
    """The coordinates, the base L0 = min(sigma_Dup, 5) and the target sigma_Dup, both read
    0.1 inside the domain of the surface and tabulated."""
    reference = target["ssvi"].extended(s.HORIZON)
    chart = Chart(
        reference.theta_times,
        reference.theta_values,
        reference.eta,
        reference.gamma,
        (0.0, s.HORIZON),
        s.TIME_FLOOR,
    )
    dupire = Dupire(target["surface"])
    lower, upper = dupire.log_moneyness_bounds
    nodes = table_times(target["surface"].expiries, s.HORIZON, **s.TABLE, required=times)

    def table(cap):
        local_volatility = Capped(dupire, cap, lower + s.DOMAIN_MARGIN, upper - s.DOMAIN_MARGIN)
        return tabulate(local_volatility, chart, nodes, s.STATE_LIMIT, s.STATE_NODES)

    return dict(chart=chart, base=table(s.BASE_CAP), target=table(None))


def particle_bounds(surface, times):
    """Forward levels of the target's digital quantiles at the nodes t_0, ..., t_{n-1}."""
    inner = times[1:-1]
    brackets = jnp.tile(jnp.asarray(s.QUANTILE_BRACKET), (inner.size, 1))
    quantiles = digital_quantiles(surface, inner, jnp.asarray(s.QUANTILES), brackets)
    return jnp.concatenate((jnp.ones((1, 2)), jnp.exp(quantiles)))


def label(degrees, iterations, steps=0):
    """The name of a model: degrees, iterations and correction steps."""
    name = f"({degrees[0]},{degrees[1]}) {iterations} iterations"
    return name if steps == 0 else f"{name} + {steps} correction steps"


def calibrate(output, backbone, quotes, target, model, times, streams):
    """Steps 3 to 6: every model of the study, by name."""
    inputs = kept(output / "leverage-inputs.npz", lambda: leverage_inputs(target, times))
    chart, base, local_volatility = inputs["chart"], inputs["base"], inputs["target"]

    @cache
    def paths():
        return simulate(model, times, streams["training"](), s.TRAINING_PATHS)

    @cache
    def correction_paths():
        return [
            simulate(model, times, streams["correction"](r), s.TRAINING_PATHS)
            for r in range(s.CORRECTION_REPLICATIONS)
        ]

    def coordinates():
        bounds = envelope(chart, times, paths().log_spot(base), **s.ENVELOPE)
        return chart.with_envelope(*bounds, time_clamp=float(times[1]))

    fitted_chart = kept(output / "coordinates.npz", coordinates)
    models = {"starting point L0": base}
    iterates = {}
    for degrees in s.DEGREES:
        basis = ChebyshevTensor(degrees)
        start = Parametrized(base, fitted_chart, basis, jnp.zeros(basis.size))
        regression = GammaRegression(basis, ridge=s.RIDGE)

        def iterate(start=start, regression=regression):
            return fixed_point(
                start, local_volatility, paths(), regression, s.ITERATIONS, s.ROW_STRIDE
            )

        chain = kept(output / f"fixed-point-{degrees[0]}-{degrees[1]}.npz", iterate)
        iterates[degrees] = [start, *chain]
        for n in s.EVALUATED_ITERATIONS:
            models[label(degrees, n)] = iterates[degrees][n]

    def particle():
        carrier = ForwardVariance(times[1:], forward_variance(target["surface"], times))
        bounds = particle_bounds(target["surface"], times)
        volatility = variance_swap_volatility(carrier, times[:-1])
        return particle_method(local_volatility, paths(), bounds, volatility)

    models["particle method"] = kept(output / "particle.npz", particle)
    support = support_points(fitted_chart, times, *s.SUPPORT)
    calls = jnp.ones(quotes.maturities.size, dtype=bool)
    for degrees in s.DEGREES:
        for n in s.CORRECTION_STARTS:
            start = iterates[degrees][n]

            def correct(start=start):
                prices = replicated_prices(
                    correction_paths(), start, quotes.maturities, quotes.log_strikes, calls
                )
                lower, upper = widened_intervals(quotes.bid, quotes.ask, quotes.liquid, prices)
                return correction_steps(
                    start,
                    correction_paths(),
                    quotes.maturities,
                    quotes.log_strikes,
                    calls,
                    lower,
                    upper,
                    quotes.weights,
                    quotes.vega,
                    trapezoid_weights(times),
                    s.CORRECTION_RADII,
                    s.CUMULATIVE_RADIUS,
                    s.LOG_LEVERAGE_BOUND,
                    support,
                )

            steps = kept(output / f"correction-{degrees[0]}-{degrees[1]}-after-{n}.npz", correct)
            for m, leverage in enumerate(steps, start=1):
                models[label(degrees, n, m)] = leverage
    return models


def corrected(models, degrees, start):
    """The name of the last correction step after start iterations; a chain ends early when a
    step cannot be solved, and without any step it is the uncorrected model."""
    for steps in range(len(s.CORRECTION_RADII), 0, -1):
        if label(degrees, start, steps) in models:
            return label(degrees, start, steps)
    return "starting point L0" if start == 0 else label(degrees, start)


def paper_models(models):
    """The models of the paper's tables and figures."""
    names = ["starting point L0", "particle method"]
    for degrees in s.DEGREES:
        names += [label(degrees, s.PAPER_TABLE_ITERATIONS), label(degrees, s.ITERATIONS)]
        names += [corrected(models, degrees, n) for n in s.CORRECTION_STARTS]
    return {name: models[name] for name in dict.fromkeys(names)}


def price(folder, models, model, times, comparison, stream, replications, paths, independent):
    """Prices (replications, quotes + comparison grid) of every model, one file per replication."""
    folder.mkdir(exist_ok=True)
    prices = []
    for r in range(replications):

        def compute(r=r):
            simulated = simulate(
                model,
                times,
                stream(r),
                paths,
                sobol=not independent,
                bridge=not independent,
                antithetic=not independent,
            )
            return {
                name: monte_carlo_prices(
                    simulated,
                    leverage,
                    comparison.maturities,
                    comparison.log_strikes,
                    comparison.is_call,
                )
                for name, leverage in models.items()
            }

        prices.append(kept(folder / f"{r:03d}.npz", compute))
    return {name: jnp.stack([p[name] for p in prices]) for name in models}


def run(pair, backbone, data, output):
    """Steps 1 to 7 for one date pair and backbone."""
    output.mkdir(parents=True, exist_ok=True)
    streams = s.streams(pair.label, pair.event_day.isoformat(), backbone)
    earlier = day_quotes(data, pair.earlier, pair.earlier_sha256)
    earlier_target = kept(output / "target-earlier-day.npz", lambda: fit_target(earlier))
    model = kept(
        output / "backbone.npz",
        lambda: fit_backbone(backbone, earlier, earlier_target["surface"], streams),
    )
    quotes = day_quotes(data, pair.event_day, pair.event_sha256)
    target = kept(output / "target.npz", lambda: fit_target(quotes))
    times = time_grid(quotes)
    models = calibrate(output, backbone, quotes, target, model, times, streams)
    comparison = Comparison(quotes, target["surface"])
    price(
        output / "evaluation",
        models,
        model,
        times,
        comparison,
        streams["evaluation"],
        s.EVALUATION_REPLICATIONS,
        s.EVALUATION_PATHS,
        independent=False,
    )
    price(
        output / "paper-evaluation",
        paper_models(models),
        model,
        times,
        comparison,
        streams["paper"],
        s.PAPER_REPLICATIONS,
        s.PAPER_PATHS,
        independent=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--pair", required=True, choices=sorted(PAIRS))
    parser.add_argument("--backbone", required=True, choices=("bergomi", "heston"))
    parser.add_argument("--data", type=Path, default=Path("data/wrds"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(PAIRS[arguments.pair], arguments.backbone, arguments.data, arguments.output)


if __name__ == "__main__":
    main()
