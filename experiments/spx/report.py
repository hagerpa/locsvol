"""The report of a run: results/<pair>/<backbone>/README.md and the figures in figures/.

The report starts with the paper's tables and figures for the run, from the 320 sets of
independent paths, and continues with every model on the 20 sets of Sobol paths: the errors and
bid-ask losses of all models, the fixed-point iteration iterate by iterate, every correction
chain, the smiles and the price errors against the bid-ask band at every quoted expiry, and the
Monte Carlo uncertainty of the implied volatilities.
"""

import numpy as np
from evaluation import jackknife
from figures import COLORS, GRID, KIND_STYLES, RC, kind_of, marker_phases, model_name, tag
from settings import (
    CORRECTION_RADII,
    CORRECTION_STARTS,
    DEGREES,
    EVALUATED_ITERATIONS,
    EVALUATION_PATHS,
    EVALUATION_REPLICATIONS,
    ITERATIONS,
    PAPER_TABLE_ITERATIONS,
)

from locsvol.pricing import implied_volatility

METRICS = (
    "Left wing",
    "Central",
    "Right wing",
    "Bid-ask loss, all quotes",
    "Bid-ask loss, liquid quotes",
)
REGIONS = ("left wing", "central", "right wing")
STEPS = len(CORRECTION_RADII)
COLUMNS = 4  # panels per row of the figures at every expiry
START_COLORS = COLORS["starts"]
START_MARKERS = {0: "o", 3: "s", 5: "D", 8: "^"}
START_DODGE = 0.08
BACKBONE_PARAMETERS = dict(
    Bergomi=("nu", "theta", "rho1", "rho2"),
    Heston=("kappa", "theta", "sigma", "rho", "v0"),
)


def cell(value, two_se=None, digits=2):
    value = float(value)
    if not np.isfinite(value):
        return "n/a"
    if two_se is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {float(two_se):.{digits}f}"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def chain(statistics, degrees, start):
    """The names of a correction chain, from its starting point to its last step."""
    first = "starting point L0" if start == 0 else model_name(degrees, start)
    steps = [model_name(degrees, start, m) for m in range(1, STEPS + 1)]
    return [first] + [name for name in steps if name in statistics]


def paired_contrast(comparison, prices, name, reference):
    """Statistics of name minus those of reference and twice their paired standard error."""
    panel = np.stack([np.asarray(prices[name]), np.asarray(prices[reference])], axis=1)
    estimate, error, _ = jackknife(
        panel, lambda p: comparison.statistics(p[..., 0, :]) - comparison.statistics(p[..., 1, :])
    )
    return np.asarray(estimate), 2 * np.asarray(error)


def pointwise_error(comparison, replication_prices):
    """Twice the standard error of the pooled implied volatility at every grid point, in bp."""
    count = comparison.count
    grid = np.asarray(replication_prices)[:, count:]
    pooled = grid.mean(axis=0)
    error = grid.std(axis=0, ddof=1) / np.sqrt(grid.shape[0])
    calls = np.ones(pooled.size, dtype=bool)
    sigma = np.asarray(implied_volatility(pooled, *comparison.grid, calls))
    shifted = np.asarray(implied_volatility(pooled + error, *comparison.grid, calls))
    return 2e4 * np.abs(shifted - sigma)


def expiry_days(comparison):
    return np.rint(365 * np.asarray(comparison.quotes.maturities)).astype(int)


def draw_report_figures(directory, comparison, pooled, statistics, increments):
    """iterations.png, corrections.png, smiles.png and spreads.png in directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(RC)
    directory.mkdir(parents=True, exist_ok=True)

    def style(kind, markevery=None, color=None):
        return {
            "color": COLORS[kind] if color is None else color,
            "linewidth": 1.0,
            "markersize": 4.0,
            "markevery": markevery,
            **KIND_STYLES[kind],
        }

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=200)
        plt.close(fig)

    # Every metric along the fixed-point iteration, with the particle method as reference.
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), layout="constrained")
    axes = axes.ravel()
    for degrees in DEGREES:
        rms = np.asarray(increments[tuple(degrees)])
        axes[0].plot(np.arange(1, rms.size + 1), rms, label=tag(degrees), **style(kind_of(degrees)))
    axes[0].set(xlabel="fixed-point iteration", ylabel="RMS coefficient increment", yscale="log")
    axes[0].set_xticks(range(0, ITERATIONS + 1, 2))
    iterations = (0, *EVALUATED_ITERATIONS)
    for metric, ax in enumerate(axes[1:]):
        for degrees in DEGREES:
            names = ["starting point L0"] + [model_name(degrees, n) for n in EVALUATED_ITERATIONS]
            estimate = np.asarray([statistics[n][0][metric] for n in names])
            error = np.asarray([statistics[n][1][metric] for n in names])
            ax.errorbar(iterations, estimate, yerr=2 * error, capsize=2, **style(kind_of(degrees)))
        ax.axhline(
            statistics["particle method"][0][metric],
            color=COLORS["particle"],
            linewidth=1.0,
            linestyle=KIND_STYLES["particle"]["linestyle"],
        )
        ax.set(xlabel="fixed-point iteration", ylabel=f"{METRICS[metric]} (bp)", yscale="log")
        ax.set_xticks(iterations)
    for ax in axes:
        ax.grid(**GRID)
    handles = [Line2D([], [], label=tag(d), **style(kind_of(d))) for d in DEGREES] + [
        Line2D(
            [],
            [],
            color=COLORS["particle"],
            linestyle=KIND_STYLES["particle"]["linestyle"],
            label="particle method",
        )
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "iterations")

    # Bid-ask losses along every correction chain.
    fig, axes = plt.subplots(2, len(DEGREES), figsize=(7.0, 4.4), layout="constrained")
    for column, degrees in enumerate(DEGREES):
        for row, metric in enumerate((3, 4)):
            ax = axes[row, column]
            for start in CORRECTION_STARTS:
                names = chain(statistics, degrees, start)
                estimate = np.asarray([statistics[n][0][metric] for n in names])
                error = np.asarray([statistics[n][1][metric] for n in names])
                offset = START_DODGE * (CORRECTION_STARTS.index(start) - 1.5)
                ax.errorbar(
                    np.arange(len(names)) + offset,
                    estimate,
                    yerr=2 * error,
                    capsize=2,
                    **{
                        **style("corrected", color=START_COLORS[start]),
                        "marker": START_MARKERS[start],
                    },
                )
            ax.set(xlabel="corrections", ylabel=f"{METRICS[metric]} (bp)", yscale="log")
            ax.set_xticks(range(STEPS + 1))
            ax.set_title(tag(degrees), loc="left")
            ax.grid(**GRID)
    handles = [
        Line2D(
            [],
            [],
            label=f"from {start} iterations",
            **{**style("corrected", color=c), "marker": START_MARKERS[start]},
        )
        for start, c in START_COLORS.items()
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    save(fig, "corrections")

    quotes, count = comparison.quotes, comparison.count
    grid_t, grid_k = (np.asarray(v) for v in comparison.grid)
    grid_days = np.rint(365 * grid_t).astype(int)
    target = np.asarray(comparison.target)
    quote_days = expiry_days(comparison)
    quote_k = np.asarray(quotes.log_strikes)
    bid, ask = np.asarray(quotes.bid), np.asarray(quotes.ask)
    liquid = np.asarray(quotes.liquid)
    scale = 1e4 / np.asarray(quotes.vega)
    band = (ask - bid) / 2 * scale
    mid_iv = np.asarray(quotes.implied_volatility)
    days = np.unique(quote_days)
    rows_of_panels = int(np.ceil(days.size / COLUMNS))
    best = model_name(DEGREES[0], PAPER_TABLE_ITERATIONS)
    curves = [
        ("particle method", "particle method", "particle"),
        *(
            (model_name(d, PAPER_TABLE_ITERATIONS), f"{tag(d)}, 5 iterations", kind_of(d))
            for d in DEGREES
        ),
        (
            _last_step(statistics, DEGREES[0], PAPER_TABLE_ITERATIONS),
            f"{tag(DEGREES[0])}, 5 iterations + corrections",
            "corrected",
        ),
    ]
    phases = marker_phases(curves)
    calls = np.ones(grid_k.size, dtype=bool)
    ivs = {
        name: np.asarray(
            implied_volatility(np.asarray(pooled[name])[count:], grid_t, grid_k, calls)
        )
        for name, _, _ in curves
    }

    def panels():
        return plt.subplots(
            rows_of_panels,
            COLUMNS,
            figsize=(7.0, 1.55 * rows_of_panels + 0.6),
            layout="constrained",
            squeeze=False,
        )

    # Implied volatility minus the target at every expiry, with the quotes' bid-ask band.
    fig, axes = panels()
    for ax, day in zip(axes.ravel(), days, strict=False):
        rows = np.flatnonzero(grid_days == day)
        k = grid_k[rows]
        rows_q = np.flatnonzero(quote_days == day)
        rows_q = rows_q[np.argsort(quote_k[rows_q])]
        centre = 1e4 * (mid_iv - np.interp(quote_k, k, target[rows]))[rows_q]
        ax.fill_between(
            quote_k[rows_q],
            centre - band[rows_q],
            centre + band[rows_q],
            color=COLORS["band"],
            linewidth=0,
            zorder=1,
        )
        ax.axhline(0.0, color=COLORS["target"], linewidth=0.8, zorder=2)
        largest = [np.percentile(np.abs(band[rows_q]), 90)]
        for name, _, kind in curves:
            residual = 1e4 * (ivs[name][rows] - target[rows])
            ax.plot(k, residual, zorder=3, **style(kind, phases[name]))
            if name == best or kind == "corrected":
                largest.append(np.nanmax(np.abs(residual)))
        limit = 1.3 * max(largest)
        ax.set_ylim(-limit, limit)
        ax.set_title(rf"$T = {day}$ days", loc="left")
        ax.grid(**GRID)
    for ax in axes.ravel()[days.size :]:
        ax.set_axis_off()
    for ax in axes.ravel()[max(days.size - COLUMNS, 0) : days.size]:
        ax.set_xlabel(r"$k = \log(K/F)$")
    fig.supylabel("implied volatility minus target (bp)", fontsize=8)
    handles = [
        Line2D([], [], color=COLORS["target"], linewidth=0.8, label="refined SVI target"),
        Patch(facecolor=COLORS["band"], linewidth=0, label="bid-ask band of the quotes"),
    ] + [Line2D([], [], label=label, **style(kind)) for _, label, kind in curves]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "smiles")

    # Prices minus mid prices over vega at every expiry, against the bid-ask band.
    fig, axes = panels()
    spread = [c for c in curves if c[2] in ("particle", "corrected")] + [
        (
            _last_step(statistics, DEGREES[1], PAPER_TABLE_ITERATIONS),
            f"{tag(DEGREES[1])}, 5 iterations + corrections",
            "fixed",
        )
    ]
    phases = marker_phases(spread)
    for ax, day in zip(axes.ravel(), days, strict=False):
        rows = np.flatnonzero(quote_days == day)
        rows = rows[np.argsort(quote_k[rows])]
        inner = rows[liquid[rows]]
        ax.fill_between(
            quote_k[rows], -band[rows], band[rows], color=COLORS["band"], linewidth=0, zorder=1
        )
        if inner.size:
            ax.fill_between(
                quote_k[inner],
                -band[inner],
                band[inner],
                color=COLORS["band_liquid"],
                linewidth=0,
                zorder=1,
            )
        ax.axhline(0, color="#4B5563", linewidth=0.5, zorder=2)
        for name, _, kind in spread:
            values = (np.asarray(pooled[name])[:count] - (bid + ask) / 2) * scale
            ax.plot(quote_k[rows], values[rows], zorder=3, **style(kind, phases[name]))
        limit = 4 * np.percentile(band[inner] if inner.size else band[rows], 90)
        ax.set_ylim(-limit, limit)
        ax.set_title(rf"$T = {day}$ days", loc="left")
        ax.grid(**GRID)
    for ax in axes.ravel()[days.size :]:
        ax.set_axis_off()
    for ax in axes.ravel()[max(days.size - COLUMNS, 0) : days.size]:
        ax.set_xlabel(r"$k = \log(K/F)$")
    fig.supylabel(r"$(P - P_{\mathrm{mid}})\,/\,\mathcal{V}$ (bp)", fontsize=8)
    handles = [
        Patch(facecolor=COLORS["band_liquid"], linewidth=0, label="bid-ask band, liquid quotes"),
        Patch(facecolor=COLORS["band"], linewidth=0, label="bid-ask band, other quotes"),
    ] + [Line2D([], [], label=label, **style(kind)) for _, label, kind in spread]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "spreads")


def _last_step(statistics, degrees, start):
    return chain(statistics, degrees, start)[-1]


def shown(name):
    """The model name as the report shows it."""
    return name.replace("+ 1 correction steps", "+ 1 correction step")


def statistic_cells(statistics, name):
    estimate, error, _ = statistics[name]
    return [cell(estimate[i], 2 * error[i]) for i in range(len(METRICS))]


def position_cells(statistics, name):
    return [cell(100 * v, digits=0) + "%" for v in statistics[name][2]]


def write_report(
    folder, pair, backbone, model, comparison, prices, statistics, target, tables_markdown
):
    """README.md of the run's folder; prices are the replication prices of every model on the
    Sobol paths, statistics their estimates, standard errors and liquid positions, target the
    bid-ask losses and liquid share inside of the target surface."""
    quotes = comparison.quotes
    days = np.unique(expiry_days(comparison))
    family = type(model).__name__
    vector = ", ".join(
        f"{name} = {float(getattr(model, name)):.4f}" for name in BACKBONE_PARAMETERS[family]
    )
    lines = [
        f"# {pair.label}, {backbone.capitalize()} backbone",
        "",
        f"{pair.event[0].upper()}{pair.event[1:]}. The {family} backbone is fitted to the S&P 500 "
        f"index options of {pair.earlier.isoformat()} ({vector}); the leverage factor is "
        f"calibrated to the {quotes.maturities.size:,} quotes of {pair.event_day.isoformat()}, "
        f"{int(np.sum(np.asarray(quotes.liquid))):,} of them liquid, at {days.size} expiries "
        f"from {days[0]} to {days[-1]} days.",
        "",
        "Contents: [the paper's tables and figures](#the-papers-tables-and-figures), "
        "[all models](#all-models), [fixed-point iteration](#fixed-point-iteration), "
        "[correction steps](#correction-steps), [smiles at every expiry](#smiles-at-every-expiry), "
        "[prices at every expiry](#prices-at-every-expiry), "
        "[Monte Carlo uncertainty](#monte-carlo-uncertainty). The numbers are in "
        "[summary.json](summary.json) and the calibrated parameters in "
        "[parameters.json](parameters.json).",
        "",
        "## The paper's tables and figures",
        "",
    ]
    paper = tables_markdown.split("\n", 2)[2].replace("## Table", "### Table")
    lines += [paper.rstrip(), ""]
    lines += [
        "![Smiles](paper/smiles.png)",
        "",
        "Implied volatilities at the expiries nearest one week, one month and one year, and their "
        "differences to the target ([PDF](paper/smiles.pdf)).",
        "",
        "![Convergence](paper/convergence.png)",
        "",
        "Root mean square coefficient increment of the fixed-point iteration, and the bid-ask loss "
        "along the correction steps of the (6,8) basis ([PDF](paper/convergence.pdf)).",
        "",
    ]

    lines += [
        "## All models",
        "",
        f"From here on every model is priced on {EVALUATION_REPLICATIONS} sets of "
        f"{EVALUATION_PATHS:,} scrambled Sobol paths. Root mean squared error of the implied "
        "volatility against the refined SVI target per region and the bid-ask losses, in "
        "volatility basis points, ± two standard errors (paired delete-one jackknife over the "
        "sets); then the share of liquid quotes inside their bid-ask interval, outside by less "
        "than two standard errors of the price (within noise), and further outside.",
        "",
    ]
    names = ["starting point L0"]
    for degrees in DEGREES:
        names += [model_name(degrees, n) for n in EVALUATED_ITERATIONS]
        names += [_last_step(statistics, degrees, start) for start in CORRECTION_STARTS]
    names.append("particle method")
    rows = [
        [shown(name), *statistic_cells(statistics, name), *position_cells(statistics, name)]
        for name in dict.fromkeys(names)
    ]
    losses, inside = target
    rows.append(
        [
            "refined SVI target",
            "",
            "",
            "",
            cell(losses["all_quotes"]),
            cell(losses["liquid_quotes"]),
        ]
        + [cell(inside, digits=0) + "%", "", ""]
    )
    lines += table(("Model", *METRICS, "Liquid inside", "Within noise", "Further outside"), rows)
    lines += [""]

    lines += [
        "## Fixed-point iteration",
        "",
        "![Iterations](figures/iterations.png)",
        "",
        "The root mean square coefficient increment of every iteration, and every metric of the "
        "iterates after 0, 3, 5, 8 and 12 iterations, ± two standard errors; the particle method "
        "as a horizontal line. Change of every iterate against the iterate after 12 iterations, "
        "± two paired standard errors:",
        "",
    ]
    rows = []
    for degrees in DEGREES:
        reference = model_name(degrees, ITERATIONS)
        for n in EVALUATED_ITERATIONS[:-1]:
            delta, two_se = paired_contrast(comparison, prices, model_name(degrees, n), reference)
            rows.append(
                [model_name(degrees, n), *(cell(d, e) for d, e in zip(delta, two_se, strict=True))]
            )
    lines += table(("Iterate", *METRICS), rows)
    lines += [""]

    lines += [
        "## Correction steps",
        "",
        "![Corrections](figures/corrections.png)",
        "",
        "Bid-ask losses along the correction chains started after 0, 3, 5 and 8 iterations, "
        "± two standard errors.",
        "",
    ]
    notes = []
    for degrees in DEGREES:
        lines += [f"### Basis {tag(degrees)}", ""]
        rows = []
        for start in CORRECTION_STARTS:
            names = chain(statistics, degrees, start)
            for steps, name in enumerate(names):
                rows.append(
                    [
                        f"{start} iterations",
                        steps,
                        *statistic_cells(statistics, name)[3:],
                        position_cells(statistics, name)[0],
                    ]
                )
            if len(names) - 1 < STEPS:
                notes.append(
                    f"The chain of {tag(degrees)} after {start} iterations ends after "
                    f"{len(names) - 1} of {STEPS} steps: the next step's optimization did not "
                    "converge."
                )
        lines += table(("Started after", "Steps", *METRICS[3:], "Liquid inside"), rows)
        lines += [""]

    lines += [
        "## Smiles at every expiry",
        "",
        "![Smiles](figures/smiles.png)",
        "",
        "Implied volatility minus the refined SVI target, in bp, with the bid-ask band of the "
        "quotes in implied volatility around their mid.",
        "",
        "## Prices at every expiry",
        "",
        "![Prices](figures/spreads.png)",
        "",
        "Model price minus mid price, divided by the vega, in bp, against the bid-ask band; the "
        "darker band marks the liquid quotes.",
        "",
        "## Monte Carlo uncertainty",
        "",
        "Twice the standard error of the pooled implied volatility at the points of the comparison "
        "grid, in bp.",
        "",
    ]
    weights = np.asarray(comparison.regions)
    rows = []
    for name in ("starting point L0", model_name(DEGREES[0], ITERATIONS), "particle method"):
        errors = pointwise_error(comparison, prices[name])
        for region, label in enumerate(REGIONS):
            values = errors[weights[region] > 0]
            values = values[np.isfinite(values)]
            rows.append(
                [
                    name,
                    label,
                    cell(np.median(values), digits=1),
                    cell(np.percentile(values, 90), digits=1),
                    cell(values.max(), digits=1),
                ]
            )
        invalid = int(np.sum(~np.isfinite(errors)))
        if invalid:
            notes.append(f"{name}: {invalid} implied volatilities of the grid are not defined.")
    lines += table(("Model", "Region", "Median", "90th percentile", "Maximum"), rows)
    if notes:
        lines += ["", "## Notes", "", *(f"- {note}" for note in notes)]
    (folder / "README.md").write_text("\n".join(lines) + "\n")
