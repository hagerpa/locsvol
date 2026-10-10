"""The results of a run: <backbone>/ beside this script.

    python experiments/essvi/publish.py --backbone heston --output <folder>

summary.json holds, for every model, the RMSE against the eSSVI per region and the central RMSE
per maturity with standard errors, from the 20 sets of Sobol paths and from the 320 sets of
independent paths; README.md the tables and the smile figure; figures/ the figure. The Results
section of README.md in this folder is rebuilt from every published run.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import parameters as p
from calibrate import backbone_model, calibrate, paper_models, synthetic_quotes, target_surface, time_grid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spx"))
import settings as s  # noqa: E402, the SPX study's settings
from evaluation import Comparison, jackknife  # noqa: E402
from results import evaluation_prices, finite, write_json  # noqa: E402, the SPX study's results.py
from run import label  # noqa: E402, the SPX study's run.py

import locsvol as lsv  # noqa: E402
from locsvol.pricing import implied_volatility  # noqa: E402

RESULTS = Path(__file__).parent  # the runs are published beside this script
REGIONS = ("left", "central", "right")


def statistics(comparison, prices):
    """Per model: regional RMSE and central RMSE per maturity in bp, with the jackknife standard
    error over the sets of paths, and the noise floor of the pooled estimate per column."""
    count, (gt, gk) = comparison.count, comparison.grid
    calls = comparison.is_call[count:]
    weights = [comparison.regions[i] for i in range(3)]
    central = np.asarray(comparison.regions[1]) > 0
    for maturity in p.MATURITIES:
        rows = central & (np.asarray(gt) == maturity)
        weights.append(jnp.asarray(rows / rows.sum()))
    weights = jnp.stack(weights)

    def statistic(pooled):
        error = implied_volatility(pooled[..., count:], gt, gk, calls) - comparison.target
        squares = jnp.where(weights > 0, error[..., None, :] ** 2 * weights, 0)
        return 1e4 * jnp.sqrt(jnp.sum(squares, axis=-1))

    names = list(prices)
    panel = jnp.stack([prices[n] for n in names], axis=1)
    estimate, error, _ = jackknife(panel, statistic)
    floors = {}
    for name in names:
        sets = np.asarray(prices[name])[:, count:]
        sigma = np.stack([np.asarray(implied_volatility(row, gt, gk, calls)) for row in sets])
        se = sigma.std(axis=0, ddof=1) / np.sqrt(sets.shape[0])
        floors[name] = [
            1e4 * float(np.sqrt(np.sum(np.where(w > 0, se**2 * w, 0)))) for w in np.asarray(weights)
        ]
    return {
        name: dict(
            estimate=[finite(v) for v in estimate[i]],
            standard_error=[finite(v) for v in error[i]],
            noise_floor=[finite(v) for v in floors[name]],
        )
        for i, name in enumerate(names)
    }


def cell(value, two_se=None):
    if value is None:
        return "n/a"
    return f"{value:.1f}" if two_se is None else f"{value:.1f} ± {two_se:.1f}"


def table(rows, title, columns):
    header = ["| Model | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    lines = [f"## {title}", "", *header]
    for name, row in rows.items():
        cells = [
            cell(e, None if se is None else 2 * se)
            for e, se in zip(row["estimate"], row["standard_error"], strict=True)
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Monte Carlo noise alone accounts for the following part of these RMSEs:", "", *header]
    for name, row in rows.items():
        lines.append(f"| {name} | " + " | ".join(cell(v) for v in row["noise_floor"]) + " |")
    return lines


def styles():
    """Line styles of the drawn models: the two bases after the paper's iterations, the particle."""
    first, second = (label(d, s.PAPER_TABLE_ITERATIONS) for d in s.DEGREES)
    return {
        first: dict(color="black", linestyle="-", marker="o", markevery=(0, 8)),
        second: dict(color="0.45", linestyle="-.", marker="s", markevery=(4, 8)),
        "particle method": dict(color="0.6", linestyle=":", marker="^", markevery=(2, 8)),
    }


def smiles(path, comparison, pooled):
    """The target smile and the models' implied volatilities, one panel per maturity."""
    count, (gt, gk) = comparison.count, comparison.grid
    t, k = np.asarray(gt), np.asarray(gk)
    calls = comparison.is_call[count:]
    target = 1e2 * np.asarray(comparison.target)
    sigma = {
        name: 1e2 * np.asarray(implied_volatility(pooled[name][count:], gt, gk, calls))
        for name in styles()
    }
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.8))
    for ax, maturity in zip(axes.ravel(), p.MATURITIES, strict=False):
        rows = t == maturity
        ax.plot(k[rows], target[rows], color="black", linestyle="--", linewidth=1, label="eSSVI target")
        for name, style in styles().items():
            ax.plot(k[rows], sigma[name][rows], linewidth=1, markersize=3, fillstyle="none", label=name, **style)
        ax.set_title(f"T = {maturity:g} y", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("k = log(K/F)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("implied volatility [%]", fontsize=8)
    legend = axes.ravel()[-1]
    legend.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    legend.legend(handles, labels, loc="center", frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=200)
    plt.close(fig)


def publish(backbone, output):
    streams = p.streams(backbone)
    surface = target_surface()
    quotes = synthetic_quotes(surface)
    ssvi = lsv.load(output / "ssvi.npz")
    times = time_grid(quotes)
    models = calibrate(output, surface, ssvi, backbone_model(backbone), times, streams)
    comparison = Comparison(quotes, surface)
    evaluation = evaluation_prices(output / "evaluation", models, s.EVALUATION_REPLICATIONS)
    chosen = paper_models(models)
    paper = evaluation_prices(output / "paper-evaluation", chosen, s.PAPER_REPLICATIONS)
    columns = [*REGIONS, *(f"{m:g} y" for m in p.MATURITIES)]
    sobol, independent = statistics(comparison, evaluation), statistics(comparison, paper)
    folder = RESULTS / backbone
    folder.mkdir(parents=True, exist_ok=True)
    write_json(
        folder / "summary.json",
        dict(
            backbone=backbone,
            columns=columns,
            sobol_evaluation=dict(
                replications=s.EVALUATION_REPLICATIONS,
                paths=s.EVALUATION_PATHS,
                segments=list(p.SEGMENTS),
                models=sobol,
            ),
            independent_evaluation=dict(
                replications=s.PAPER_REPLICATIONS, paths=s.PAPER_PATHS, models=independent
            ),
        ),
    )
    pooled = {name: jnp.mean(paper[name], axis=0) for name in chosen}
    smiles(folder / "figures" / "smiles.png", comparison, pooled)
    lines = [
        f"# The parametric target with the {backbone.capitalize()} backbone",
        "",
        "Root mean squared error of the implied volatility against the eSSVI surface, in volatility",
        "basis points, over the left wing, the central region and the right wing, and over the central",
        "region of each maturity; point estimates ± two standard errors (paired delete-one jackknife",
        "over the sets of paths).",
        "",
        *table(independent, f"On {s.PAPER_REPLICATIONS} sets of {s.PAPER_PATHS:,} independent paths", columns),
        "",
        *table(sobol, f"On {s.EVALUATION_REPLICATIONS} sets of {s.EVALUATION_PATHS:,} Sobol paths", columns),
        "",
        "## Smiles",
        "",
        "![smiles](figures/smiles.png)",
        "",
        f"Implied volatilities of the models on the {s.PAPER_REPLICATIONS} sets of independent paths",
        "at the five maturities, with the eSSVI target.",
        "",
    ]
    (folder / "README.md").write_text("\n".join(lines))
    update_readme(RESULTS)


def results_section(results):
    """The Results section of README.md, from the summaries of the published runs."""
    chosen = label(s.DEGREES[0], s.PAPER_TABLE_ITERATIONS)
    lines = [
        "## Results",
        "",
        f"Each folder `<backbone>/` holds the results of one run of `calibrate.py`. Left wing, central",
        "region and right wing RMSE against the eSSVI surface, in volatility basis points, on",
        f"{s.PAPER_REPLICATIONS} sets of {s.PAPER_PATHS:,} independent paths:",
        "",
        f"| Backbone | RMSE, {chosen} | RMSE, particle method |",
        "|---|---|---|",
    ]
    for path in sorted(results.glob("*/summary.json")):
        record = json.loads(path.read_text())
        rows = record["independent_evaluation"]["models"]

        def rmse(row):
            return " / ".join(cell(v) for v in row["estimate"][:3])

        name = record["backbone"]
        lines.append(f"| [{name.capitalize()}]({name}) | {rmse(rows[chosen])} | {rmse(rows['particle method'])} |")
    lines += [
        "",
        "Each folder opens with its report, `README.md`: the tables of the regional and the",
        "per-maturity errors on the independent and on the Sobol paths and the smile figure; beside",
        "it `summary.json` with every model's errors and standard errors and `figures/`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def update_readme(results):
    """Rewrite the Results section of README.md, up to the next section."""
    path = results / "README.md"
    text = path.read_text()
    match = re.search(r"^## Results\n.*?(?=^## )", text, flags=re.S | re.M)
    if match is None:
        raise ValueError("README.md has no Results section")
    path.write_text(text[: match.start()] + results_section(results) + text[match.end() :])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--backbone", required=True, choices=("bergomi", "heston"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    matplotlib.use("Agg")
    publish(arguments.backbone, arguments.output)


if __name__ == "__main__":
    main()
