"""The results of a run: <pair>/<backbone>/ beside this script.

    python experiments/spx/results.py --pair fomc --backbone bergomi --output <folder>

summary.json holds, for every model, the RMSE against the target per region, the bid-ask losses
and the shares of liquid quotes inside their intervals, with standard errors, from the 20 sets of
Sobol paths; parameters.json the fitted backbone, the target, the coordinates and the coefficients
of every parametrized leverage factor. paper/ holds the paper's tables (tables.md, tables.json) and
figures (PDF and PNG) from the 320 sets of independent paths, and README.md the report of the run
with the figures of figures/ (report.py). The Results section of README.md in this folder is
rebuilt from every published run.
"""

import argparse
import importlib.metadata
import json
import math
import re
import subprocess
from dataclasses import fields, is_dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import settings as s
from date_pairs import PAIRS
from evaluation import Comparison, jackknife
from figures import draw
from report import draw_report_figures, write_report
from run import calibrate, corrected, day_quotes, label, paper_models, time_grid

import locsvol as lsv
from locsvol.calibration.gauss_newton import interval_residuals
from locsvol.leverage import Parametrized
from locsvol.pricing import black_price, implied_volatility

RESULTS = Path(__file__).parent  # the runs are published beside this script
REGIONS = ("left", "central", "right")
LOSSES = ("all_quotes", "liquid_quotes")
POSITIONS = ("inside", "within_noise", "clearly_outside")


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def plain(value):
    """JSON form of numbers, arrays, tuples and dataclasses; non-finite numbers become null."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple | list):
        return [plain(item) for item in value]
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return finite(array)
    return [plain(item) for item in array]


def describe(name):
    """Method, degrees, iterations and correction steps of a model name of run.py."""
    if name == "starting point L0":
        return dict(method="fixed-point iteration", degrees=None, iterations=0, correction_steps=0)
    if name == "particle method":
        return dict(method="particle method", degrees=None, iterations=None, correction_steps=None)
    p, q, n, steps = re.fullmatch(
        r"\((\d+),(\d+)\) (\d+) iterations(?: \+ (\d+) correction steps)?", name
    ).groups()
    return dict(
        method="correction steps" if steps else "fixed-point iteration",
        degrees=[int(p), int(q)],
        iterations=int(n),
        correction_steps=int(steps or 0),
    )


def evaluation_prices(folder, names, replications):
    """Prices (replications, quotes + grid) of the models from the files of run.py."""
    files = [lsv.load(folder / f"{r:03d}.npz") for r in range(replications)]
    return {name: jnp.stack([f[name] for f in files]) for name in names}


def model_statistics(comparison, prices):
    """Estimates, standard errors and liquid positions of every model, paired over replications."""
    names = list(prices)
    panel = jnp.stack([prices[name] for name in names], axis=1)
    estimate, error, _ = jackknife(panel, comparison.statistics)
    positions = comparison.positions(panel)
    return (
        {
            name: (np.asarray(estimate[i]), np.asarray(error[i]), np.asarray(positions[i]))
            for i, name in enumerate(names)
        },
        jnp.mean(panel, axis=0),
        names,
    )


def target_statistics(comparison, surface):
    """Bid-ask losses and liquid quotes inside for the target surface's own Black prices."""
    quotes = comparison.quotes
    calls = jnp.ones(quotes.maturities.size, dtype=bool)
    prices = black_price(
        surface.total_variance(quotes.maturities, quotes.log_strikes), quotes.log_strikes, calls
    )
    losses = [
        1e4 * jnp.linalg.norm(interval_residuals(prices, quotes.bid, quotes.ask, w, quotes.vega))
        for w in (quotes.weights, quotes.liquid_weights)
    ]
    prices, bid, ask = (np.asarray(v) for v in (prices, quotes.bid, quotes.ask))
    inside = ((prices >= bid) & (prices <= ask))[np.asarray(quotes.liquid)]
    return dict(zip(LOSSES, map(finite, losses), strict=True)), finite(100 * inside.mean())


def summary(pair, backbone, statistics, target):
    rows = []
    for name, (estimate, error, positions) in statistics.items():
        rows.append(
            dict(
                label=name,
                **describe(name),
                target_rmse_bp=dict(zip(REGIONS, map(finite, estimate[:3]), strict=True)),
                target_rmse_standard_error_bp=dict(
                    zip(REGIONS, map(finite, error[:3]), strict=True)
                ),
                bid_ask_loss_bp=dict(zip(LOSSES, map(finite, estimate[3:]), strict=True)),
                bid_ask_loss_standard_error_bp=dict(
                    zip(LOSSES, map(finite, error[3:]), strict=True)
                ),
                liquid_quotes_percent=dict(
                    zip(POSITIONS, map(finite, 100 * positions), strict=True)
                ),
            )
        )
    losses, inside = target
    rows.append(
        dict(
            label="target surface",
            method="refined SVI target",
            degrees=None,
            iterations=None,
            correction_steps=None,
            bid_ask_loss_bp=losses,
            liquid_quotes_percent=dict(inside=inside),
        )
    )
    return dict(
        date_pair=dict(
            label=pair.label,
            earlier_day=pair.earlier.isoformat(),
            event_day=pair.event_day.isoformat(),
            event=pair.event,
        ),
        backbone=backbone,
        evaluation=dict(
            replications=s.EVALUATION_REPLICATIONS,
            paths_per_replication=s.EVALUATION_PATHS,
            paths="scrambled Sobol points with a Brownian bridge construction",
            standard_errors="paired delete-one jackknife over the replications",
        ),
        models=rows,
    )


def parameters(pair, model, target, coordinates, models):
    """Backbone, target, coordinates and coefficients, with the code and data they come from."""
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True
    ).stdout.strip()
    coefficients = [
        dict(label=name, **describe(name), coefficients=plain(leverage.coefficients))
        for name, leverage in models.items()
        if isinstance(leverage, Parametrized)
    ]
    return dict(
        backbone=dict(family=type(model).__name__, parameters=plain(model)),
        target=dict(ssvi=plain(target["ssvi"]), refined_svi=plain(target["surface"])),
        coordinates=plain(coordinates),
        leverage=dict(
            base="Dupire local volatility of the target, read at x clipped to the quantiles of the "
            "SSVI fit, tabulated",
            basis="Chebyshev tensor, time-outer: coefficient i(q+1)+j multiplies T_i(tau) T_j(u)",
            models=coefficients,
        ),
        provenance=dict(
            commit=commit or None,
            earlier_day_sha256=pair.earlier_sha256,
            event_day_sha256=pair.event_sha256,
            versions={n: importlib.metadata.version(n) for n in ("jax", "numpy", "scipy")},
        ),
    )


def increments(output):
    """RMS coefficient increments of the iterations, for each basis."""
    result = {}
    for degrees in s.DEGREES:
        chain = lsv.load(output / f"fixed-point-{degrees[0]}-{degrees[1]}.npz")
        beta = np.stack(
            [np.zeros_like(np.asarray(chain[0].coefficients))]
            + [np.asarray(c.coefficients) for c in chain]
        )
        result[tuple(degrees)] = np.sqrt(np.mean(np.diff(beta, axis=0) ** 2, axis=1))
    return result


def noise_floor(comparison, replication_prices):
    """Monte Carlo noise of the regional RMSE: the region-weighted root mean square of the
    implied-volatility standard errors, in bp."""
    count = comparison.count
    grid = np.asarray(replication_prices)[..., count:]
    pooled = grid.mean(axis=0)
    error = grid.std(axis=0, ddof=1) / np.sqrt(grid.shape[0])
    calls = comparison.is_call[count:]
    sigma = np.asarray(implied_volatility(pooled, *comparison.grid, calls))
    shifted = np.asarray(implied_volatility(pooled + error, *comparison.grid, calls))
    squares = (shifted - sigma) ** 2
    return [
        1e4 * float(np.sqrt(np.sum(np.where(w > 0, squares * w, 0))))
        for w in map(np.asarray, comparison.regions)
    ]


def cell(value, two_se=None, digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if two_se is None else f"{value:.{digits}f} ± {two_se:.{digits}f}"


def paper_tables(comparison, prices, target):
    """Tables 1 and 2 of the paper from the paper's evaluation."""
    statistics, _, _ = model_statistics(comparison, prices)
    n = s.PAPER_TABLE_ITERATIONS
    first = [
        (f"Joint regression ({d[0]},{d[1]}), {n} iterations", f"({d[0]},{d[1]}) {n} iterations")
        for d in s.DEGREES
    ]
    first.append(("Particle method", "particle method"))
    table_1 = []
    for title, name in first:
        estimate, error, _ = statistics[name]
        table_1.append(
            dict(
                label=title,
                rmse_bp=dict(zip(REGIONS, map(finite, estimate[:3]), strict=True)),
                two_standard_errors_bp=dict(zip(REGIONS, map(finite, 2 * error[:3]), strict=True)),
                noise_floor_bp=dict(
                    zip(REGIONS, noise_floor(comparison, prices[name]), strict=True)
                ),
            )
        )
    table_2 = []
    for d in s.DEGREES:
        for start in s.CORRECTION_STARTS:
            name = corrected(prices, d, start)
            estimate, error, positions = statistics[name]
            table_2.append(
                dict(
                    degrees=f"({d[0]},{d[1]})",
                    label=f"{start} iterations",
                    bid_ask_loss_bp=dict(zip(LOSSES, map(finite, estimate[3:]), strict=True)),
                    two_standard_errors_bp=dict(
                        zip(LOSSES, map(finite, 2 * error[3:]), strict=True)
                    ),
                    liquid_inside_percent=finite(100 * positions[0]),
                )
            )
    estimate, error, positions = statistics["particle method"]
    table_2.append(
        dict(
            degrees=None,
            label="Particle method",
            bid_ask_loss_bp=dict(zip(LOSSES, map(finite, estimate[3:]), strict=True)),
            two_standard_errors_bp=dict(zip(LOSSES, map(finite, 2 * error[3:]), strict=True)),
            liquid_inside_percent=finite(100 * positions[0]),
        )
    )
    losses, inside = target
    table_2.append(
        dict(
            degrees=None,
            label="Refined SVI target",
            bid_ask_loss_bp=losses,
            two_standard_errors_bp=None,
            liquid_inside_percent=inside,
        )
    )
    return dict(
        evaluation=dict(
            replications=s.PAPER_REPLICATIONS,
            paths_per_replication=s.PAPER_PATHS,
            paths="independent pseudo-random paths",
        ),
        table_1=table_1,
        table_2=table_2,
    )


def tables_markdown(tables):
    """The paper's two tables as Markdown."""
    e = tables["evaluation"]
    lines = [
        "# Tables of the paper",
        "",
        f"Both tables are computed on {e['replications']} replications of "
        f"{e['paths_per_replication']:,} {e['paths']}. Entries are point estimates ± two "
        "standard errors (paired delete-one jackknife over the replications), in volatility "
        "basis points.",
        "",
        "## Table 1",
        "",
        "Root mean squared error (RMSE) of the implied volatility against the refined SVI target "
        f"after {s.PAPER_TABLE_ITERATIONS} fixed-point iterations, with the particle method as a "
        "reference.",
        "",
        "| Method | Left | Central | Right |",
        "|---|---|---|---|",
    ]
    for row in tables["table_1"]:
        cells = [cell(row["rmse_bp"][r], row["two_standard_errors_bp"][r]) for r in REGIONS]
        lines.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    lines += ["", "Monte Carlo noise alone accounts for the following part of these RMSEs:", ""]
    lines += ["| Method | Left | Central | Right |", "|---|---|---|---|"]
    for row in tables["table_1"]:
        lines.append(
            f"| {row['label']} | "
            + " | ".join(cell(row["noise_floor_bp"][r]) for r in REGIONS)
            + " |"
        )
    lines += [
        "",
        "## Table 2",
        "",
        "Bid-ask loss and share of liquid quotes inside their bid-ask interval after the "
        "correction steps.",
        "",
        "| Degrees | Correction started after | All quotes | Liquid quotes | Liquid inside |",
        "|---|---|---|---|---|",
    ]
    for row in tables["table_2"]:
        two_se = row["two_standard_errors_bp"] or {}
        cells = [cell(row["bid_ask_loss_bp"][key], two_se.get(key)) for key in LOSSES]
        degrees = row["degrees"] or row["label"]
        start = row["label"] if row["degrees"] else ""
        inside = cell(row["liquid_inside_percent"], digits=0)
        lines.append(f"| {degrees} | {start} | " + " | ".join(cells) + f" | {inside}% |")
    return "\n".join(lines) + "\n"


def overview(summaries):
    """The README table over all published runs, ordered by the event day."""
    n = s.PAPER_TABLE_ITERATIONS
    degrees = [f"({d[0]},{d[1]})" for d in s.DEGREES]
    lines = [
        "| Date pair | Backbone | "
        + " | ".join(f"RMSE, {d} after {n} iterations" for d in degrees)
        + " | RMSE, particle method | "
        + " | ".join(f"Liquid inside, {d} corrected" for d in degrees)
        + " | Liquid inside, particle method |",
        "|---|---|" + "---|" * (2 * len(degrees) + 2),
    ]
    for record in sorted(summaries, key=lambda r: (r["date_pair"]["event_day"], r["backbone"])):
        rows = {row["label"]: row for row in record["models"]}
        fixed = [rows[label(d, n)] for d in s.DEGREES]
        last = [rows[corrected(rows, d, n)] for d in s.DEGREES]
        particle = rows["particle method"]

        def rmse(row):
            return " / ".join(cell(row["target_rmse_bp"][r], digits=1) for r in REGIONS)

        def inside(row):
            return f"{cell(row['liquid_quotes_percent']['inside'], digits=0)}%"

        pair = record["date_pair"]
        folder = f"{pair['label']}/{record['backbone']}"
        lines.append(
            f"| [`{pair['label']}`]({folder}) ({pair['earlier_day']} → {pair['event_day']}) "
            f"| [{record['backbone'].capitalize()}]({folder}) | "
            + " | ".join(rmse(row) for row in fixed)
            + f" | {rmse(particle)} | "
            + " | ".join(inside(row) for row in last)
            + f" | {inside(particle)} |"
        )
    return lines


def results_section(summaries):
    """The Results section of README.md, from the summaries of the published runs."""
    events = {r["date_pair"]["label"]: r["date_pair"]["event"] for r in summaries}
    lines = [
        "## Results",
        "",
        "Each folder `<pair>/<backbone>/` holds the results of one run of `run.py`: the backbone "
        "(Bergomi or Heston) is fitted to the S&P 500 index options of the earlier day and the "
        "leverage factor to those of the event day. Every run is evaluated on "
        f"{s.EVALUATION_REPLICATIONS} sets of {s.EVALUATION_PATHS:,} scrambled Sobol paths.",
        "",
        *overview(summaries),
        "",
        "The RMSE columns give the root mean squared difference between model and target implied "
        "volatilities over the left wing, the central region and the right wing, in volatility "
        "basis points, for both bases after 5 iterations and for the particle method. The inside "
        "columns give the share of liquid quotes inside their bid-ask interval for both bases "
        "after four correction steps started after 5 iterations, and for the particle method.",
        "",
        *(f"- `{label}`: {event}" for label, event in sorted(events.items())),
        "",
        "Each folder opens with its report, `README.md`: the paper's tables and figures for that "
        "date pair and backbone, every model's errors and bid-ask losses, the fixed-point "
        "iteration iterate by iterate, every correction chain, and the smiles and prices at every "
        "expiry. Besides it:",
        "",
        "- `paper/`: the paper's tables (`tables.md`, `tables.json`) and figures (PDF and PNG), "
        f"from {s.PAPER_REPLICATIONS} sets of {s.PAPER_PATHS:,} independent paths.",
        "- `summary.json`: for every model (the starting point L0, the fixed-point iterates after "
        "3, 5, 8 and 12 iterations of both bases, every correction step and the particle method) "
        "the RMSE per region and the bid-ask loss over all and over the liquid quotes, each with "
        "its standard error, and the shares of liquid quotes inside, within Monte Carlo noise of "
        "and clearly outside their intervals; also the bid-ask statistics of the target surface.",
        "- `parameters.json`: the fitted backbone, the SSVI and refined SVI target, the "
        "coordinates and the coefficients of every parametrized leverage factor, with the code "
        "commit and the SHA-256 of both data files.",
        "- `figures/`: the figures of the report.",
        "",
    ]
    return "\n".join(lines) + "\n"


def update_readme(summaries):
    """Rewrite the Results section of README.md, up to the next section."""
    path = RESULTS / "README.md"
    text = path.read_text()
    match = re.search(r"^## Results\n.*?(?=^## )", text, flags=re.S | re.M)
    if match is None:
        raise ValueError("README.md has no Results section")
    path.write_text(text[: match.start()] + results_section(summaries) + text[match.end() :])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def publish(pair, backbone, data, output):
    """Write the results of one finished run."""
    streams = s.streams(pair.label, pair.event_day.isoformat(), backbone)
    quotes = day_quotes(data, pair.event_day, pair.event_sha256)
    target = lsv.load(output / "target.npz")
    model = lsv.load(output / "backbone.npz")
    times = time_grid(quotes)
    models = calibrate(output, backbone, quotes, target, model, times, streams)
    comparison = Comparison(quotes, target["surface"])
    prices = evaluation_prices(output / "evaluation", models, s.EVALUATION_REPLICATIONS)
    statistics, pooled, names = model_statistics(comparison, prices)
    folder = RESULTS / pair.label / backbone
    folder.mkdir(parents=True, exist_ok=True)
    target_row = target_statistics(comparison, target["surface"])
    write_json(folder / "summary.json", summary(pair, backbone, statistics, target_row))
    coordinates = lsv.load(output / "leverage-inputs.npz")["chart"]
    write_json(folder / "parameters.json", parameters(pair, model, target, coordinates, models))
    estimates = {name: (statistics[name][0], statistics[name][1]) for name in names}
    pooled_by_name = {name: pooled[i] for i, name in enumerate(names)}
    draw_report_figures(
        folder / "figures", comparison, pooled_by_name, statistics, increments(output)
    )
    chosen = paper_models(models)
    paper_prices = evaluation_prices(output / "paper-evaluation", chosen, s.PAPER_REPLICATIONS)
    tables = paper_tables(comparison, paper_prices, target_row)
    (folder / "paper").mkdir(exist_ok=True)
    write_json(folder / "paper" / "tables.json", tables)
    (folder / "paper" / "tables.md").write_text(tables_markdown(tables))
    paper_pooled = {name: jnp.mean(p, axis=0) for name, p in paper_prices.items()}
    for formats, dpi in ((("pdf",), "figure"), (("png",), 200)):
        draw(
            folder / "paper",
            comparison,
            paper_pooled,
            estimates,
            increments(output),
            formats,
            dpi,
            ("smiles", "convergence"),
        )
    write_report(
        folder,
        pair,
        backbone,
        model,
        comparison,
        prices,
        statistics,
        target_row,
        tables_markdown(tables),
    )
    summaries = [json.loads(path.read_text()) for path in sorted(RESULTS.glob("*/*/summary.json"))]
    update_readme(summaries)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--pair", required=True, choices=sorted(PAIRS))
    parser.add_argument("--backbone", required=True, choices=("bergomi", "heston"))
    parser.add_argument("--data", type=Path, default=Path("data/wrds"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    publish(PAIRS[arguments.pair], arguments.backbone, arguments.data, arguments.output)


if __name__ == "__main__":
    main()
