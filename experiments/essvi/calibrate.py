"""The parametric study: the eSSVI surface of the light-speed paper as target, for one backbone.

    python experiments/essvi/calibrate.py --backbone heston --output <folder>

The steps of the SPX study (experiments/spx/run.py) with the parametric target in place of the
market quotes and the fitted surfaces: no backbone fit (the backbones are the reference models of
Beiglboeck, Hasenbichler and Pammer), no correction steps (there are no bid-ask intervals), the
maturities 0.25 to 3 years and the Sobol points in segments of a year (parameters.py). The
quotes are the eSSVI's implied volatilities on the comparison band; the coordinates come from
the SSVI fit to them, and the target local volatility is the eSSVI's own Dupire local volatility.
"""

import argparse
import sys
from functools import cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import parameters as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spx"))
import settings as s  # noqa: E402, the SPX study's settings
from evaluation import Comparison  # noqa: E402
from quotes import Quotes, block_weights  # noqa: E402
from run import kept, label, particle_bounds, price  # noqa: E402, the SPX study's run.py

from locsvol.backbones import Bergomi, ForwardVariance, Heston  # noqa: E402
from locsvol.calibration import fixed_point, particle_method, variance_swap_volatility  # noqa: E402
from locsvol.leverage import (  # noqa: E402
    Chart,
    Clipped,
    Parametrized,
    quantile_bounds,
    quantile_envelope,
    table_times,
    tabulate,
)
from locsvol.pricing import black_price, black_vega  # noqa: E402
from locsvol.regression import ChebyshevTensor, GammaRegression  # noqa: E402
from locsvol.simulation import graded_times, simulate, with_dates  # noqa: E402
from locsvol.targets import ESSVI, Dupire, fit_ssvi, forward_variance  # noqa: E402


def target_surface():
    """The eSSVI surface, with theta continued linearly beyond 3 years to the horizon."""
    tenors, atm = np.asarray(p.MATURITIES), np.asarray(p.ATM_VOLATILITY)
    surface = ESSVI(
        jnp.asarray(np.concatenate(([0.0], tenors))),
        jnp.asarray(np.concatenate(([0.0], atm**2 * tenors))),
        p.RHO_0,
        p.RHO_M,
        p.A,
        p.ETA,
        p.LAMBDA,
    )
    return surface.extended(p.HORIZON + 0.15)


def synthetic_quotes(surface):
    """The surface's implied volatilities at STRIKES log strikes per maturity on the band of the
    SPX study within RANGE, as quotes without spread (bid = ask = mid)."""
    maturities, strikes = [], []
    for maturity in p.MATURITIES:
        half = s.BAND[0] + s.BAND[1] * maturity
        lower, upper = max(-half, p.RANGE[0]), min(half, p.RANGE[1])
        maturities += [maturity] * p.STRIKES
        strikes += np.linspace(lower, upper, p.STRIKES).tolist()
    t, k = jnp.asarray(maturities), jnp.asarray(strikes)
    w = surface.total_variance(t, k)
    sigma = jnp.sqrt(w / t)
    calls = jnp.ones(t.size, dtype=bool)
    mid = black_price(w, k, calls)
    puts = k < 0.0
    weights = block_weights(maturities, np.asarray(puts).tolist())
    ones = jnp.ones(t.size)
    return Quotes(
        t,
        k,
        ones,
        ones,
        mid,
        mid,
        mid,
        puts,
        sigma,
        black_vega(sigma, t, k),
        jnp.zeros(t.size),
        ones,
        jnp.ones(t.size, dtype=bool),
        weights,
        weights,
    )


def time_grid(quotes):
    """The graded grid of the SPX study, continued to the horizon of this study."""
    times = graded_times(p.HORIZON, s.GRADED_HORIZON, s.GRADED_STEPS, s.GRADING_POWER, s.TAIL_STEP)
    return with_dates(times, quotes.maturities)


def backbone_model(backbone):
    """The reference Heston or one-factor Bergomi model of the light-speed paper."""
    if backbone == "heston":
        return Heston(**p.HESTON, floor=s.HESTON_FLOOR)
    xi = ForwardVariance(
        jnp.asarray(p.FORWARD_VARIANCE_BOUNDARIES), jnp.asarray(p.FORWARD_VARIANCE)
    )
    return Bergomi(**p.BERGOMI, floor=s.BERGOMI_FIXED["floor"], forward_variance=xi)


def leverage_inputs(surface, ssvi, times):
    """As in the SPX study: the coordinates from the SSVI fit, the target sigma_Dup of the eSSVI
    read at x clipped to the SSVI quantiles and tabulated; the leverage factors start from L0."""
    reference = ssvi.extended(p.HORIZON)
    chart = Chart(
        reference.theta_times,
        reference.theta_values,
        reference.eta,
        reference.gamma,
        (0.0, p.HORIZON),
        s.TIME_FLOOR,
    )
    nodes = table_times(jnp.asarray(p.MATURITIES), p.HORIZON, **s.TABLE, required=times)
    level = s.PATHS_BEYOND_EDGE / s.TRAINING_PATHS
    bounds = quantile_bounds(reference, nodes, level)
    table = tabulate(Clipped(Dupire(surface), *bounds), chart, nodes, s.STATE_LIMIT, s.STATE_NODES)
    envelope = quantile_envelope(chart, reference, nodes, level)
    return dict(chart=chart.with_envelope(*envelope, time_clamp=float(times[1])), target=table)


def calibrate(output, surface, ssvi, model, times, streams):
    """Every model of the study, by name: L0, the iterates of both bases, the particle method."""
    inputs = kept(output / "leverage-inputs.npz", lambda: leverage_inputs(surface, ssvi, times))
    chart, local_volatility = inputs["chart"], inputs["target"]

    @cache
    def paths():
        return simulate(model, times, streams["training"](), s.TRAINING_PATHS, segments=p.SEGMENTS)

    models = {"starting point L0": local_volatility}
    for degrees in s.DEGREES:
        basis = ChebyshevTensor(degrees)
        start = Parametrized(local_volatility, chart, basis, jnp.zeros(basis.size))
        regression = GammaRegression(basis, ridge=s.RIDGE)

        def iterate(start=start, regression=regression):
            return fixed_point(
                start, local_volatility, paths(), regression, s.ITERATIONS, s.ROW_STRIDE
            )

        chain = kept(output / f"fixed-point-{degrees[0]}-{degrees[1]}.npz", iterate)
        for n in s.EVALUATED_ITERATIONS:
            models[label(degrees, n)] = chain[n - 1]

    def particle():
        carrier = ForwardVariance(times[1:], forward_variance(surface, times))
        volatility = variance_swap_volatility(carrier, times[:-1])
        return particle_method(
            local_volatility, paths(), particle_bounds(surface, times), volatility
        )

    models["particle method"] = kept(output / "particle.npz", particle)
    return models


def paper_models(models):
    """The models of the study's tables and figure."""
    names = ["starting point L0", "particle method"]
    for degrees in s.DEGREES:
        names += [label(degrees, s.PAPER_TABLE_ITERATIONS), label(degrees, s.ITERATIONS)]
    return {name: models[name] for name in names}


def run(backbone, output):
    output.mkdir(parents=True, exist_ok=True)
    streams = p.streams(backbone)
    surface = target_surface()
    quotes = synthetic_quotes(surface)
    data = (quotes.maturities, quotes.log_strikes, quotes.implied_volatility, quotes.weights)
    ssvi = kept(output / "ssvi.npz", lambda: fit_ssvi(*data))
    times = time_grid(quotes)
    model = backbone_model(backbone)
    models = calibrate(output, surface, ssvi, model, times, streams)
    comparison = Comparison(quotes, surface)
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
        segments=p.SEGMENTS,
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
    parser.add_argument("--backbone", required=True, choices=("bergomi", "heston"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.backbone, arguments.output)


if __name__ == "__main__":
    main()
