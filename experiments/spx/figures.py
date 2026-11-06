"""The figures of a run, drawn as in the paper.

smiles: at the expiries nearest one week, one month and one year, the implied volatilities of the
target, of the starting point L0, of the particle method and of the fixed point after five
iterations for both bases; below, the differences to the target. convergence: the root mean
square change of the leverage factor along the simulated paths in every iteration, and the
bid-ask loss over all quotes along the correction steps of the (6,8) basis. spreads: the model
prices minus the mid prices, divided by the vega, for the particle method, the (6,8) basis after
twelve iterations and after five iterations and four correction steps, against the bid-ask band.
Every model keeps one colour, line style and marker, so that it is recognized without colour.
"""

import numpy as np
from run import corrected
from settings import CORRECTION_RADII, CORRECTION_STARTS, DEGREES, ITERATIONS

from locsvol.pricing import implied_volatility

PANELS = ((7, "1W"), (30, "1M"), (365, "1Y"))  # target calendar days of the panels
FIGURE_DEGREES = (6, 8)
SMILE_ITERATIONS = 5
COLORS = dict(
    quotes="#A9A9A9",
    band="#F2F2F2",
    band_liquid="#DEDEDE",
    band_edge="#9A9A9A",
    target="#1B1B1B",
    backbone="#8C8C8C",
    particle="#59A14F",
    fixed="#F28E2B",
    fixed_other="#B5560A",
    corrected="#4E79A7",
    starts={0: "#8C8C8C", 3: "#A0CBE8", 5: "#4E79A7", 8: "#2B4C7E"},
)
# The figures are printed at 0.9 of the text width of the SIAM class (444.5 pt), so they are
# drawn at that width and their text at about the size of the document's footnotes.
WIDTH = 0.9 * 444.53 / 72.27

RC = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#3C3C3C",
    "axes.linewidth": 0.6,
    "axes.labelcolor": "#222222",
    "axes.titlesize": 9,
    "axes.axisbelow": True,
    "xtick.color": "#3C3C3C",
    "ytick.color": "#3C3C3C",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 8,
}
GRID = dict(axis="y", color="#E4E4E4", linewidth=0.5)
KIND_STYLES = {
    "backbone": dict(linestyle=":"),
    "particle": dict(linestyle="-.", marker="^", markeredgecolor="white", markeredgewidth=0.5),
    "fixed": dict(linestyle="--", marker="D", markerfacecolor="white", markersize=3.6),
    "fixed_other": dict(
        linestyle="-", marker="D", markeredgecolor="white", markeredgewidth=0.5, markersize=3.6
    ),
    "corrected": dict(linestyle="-", marker="o", markeredgecolor="white", markeredgewidth=0.5),
}
MARKER_SPACING = 0.1  # between the markers of one curve, as a fraction of the axes diagonal
START_DODGE = 0.1  # horizontal offset between the correction curves of consecutive starts


def tag(degrees):
    return f"({degrees[0]},{degrees[1]})"


def model_name(degrees, iterations, steps=0):
    """The name of a model in run.py."""
    name = f"{tag(degrees)} {iterations} iterations"
    return name if steps == 0 else f"{name} + {steps} correction steps"


def kind_of(degrees):
    return "fixed" if tuple(degrees) == FIGURE_DEGREES else "fixed_other"


def marker_phases(rows):
    """Marker positions that alternate between curves, so that coinciding curves stay visible."""
    marked = [name for name, _, kind in rows if "marker" in KIND_STYLES[kind]]
    return {
        name: (MARKER_SPACING * position / len(marked), MARKER_SPACING)
        for position, name in enumerate(marked)
    }


def panel_days(maturities):
    """The quoted expiries, in days, nearest one week, one month and one year."""
    days = np.unique(np.rint(365 * np.asarray(maturities)).astype(int))
    return [(int(days[np.argmin(np.abs(days - target))]), label) for target, label in PANELS]


def draw(
    directory,
    comparison,
    prices,
    statistics,
    increments,
    formats=("png",),
    dpi=300,
    figures=("smiles", "convergence", "spreads"),
):
    """Draw the figures into directory.

    prices maps model names to pooled prices at the quotes and the comparison grid, statistics
    maps them to the estimates and standard errors of Comparison.statistics, and increments maps
    the degrees to the root mean square changes of the leverage factor along the simulated paths
    in every iteration.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(RC)
    directory.mkdir(parents=True, exist_ok=True)
    quotes, count = comparison.quotes, comparison.count
    grid_t, grid_k = (np.asarray(v) for v in comparison.grid)
    calls = np.ones(grid_k.size, dtype=bool)
    target = 100 * np.asarray(comparison.target)
    bid, ask = np.asarray(quotes.bid), np.asarray(quotes.ask)
    liquid = np.asarray(quotes.liquid)
    scale = 1e4 / np.asarray(quotes.vega)
    band = (ask - bid) / 2 * scale
    grid_days = np.rint(365 * grid_t).astype(int)
    quote_days = np.rint(365 * np.asarray(quotes.maturities)).astype(int)
    quote_k = np.asarray(quotes.log_strikes)
    days = panel_days(grid_t)

    def save(fig, name):
        if name in figures:
            for suffix in formats:
                metadata = {"CreationDate": None} if suffix == "pdf" else None
                fig.savefig(directory / f"{name}.{suffix}", dpi=dpi, metadata=metadata)
        plt.close(fig)

    def style(kind, markevery=None, color=None):
        return {
            "color": COLORS[kind] if color is None else color,
            "linewidth": 1.0,
            "markersize": 4.0,
            "markevery": markevery,
            **KIND_STYLES[kind],
        }

    def iv(name):
        price = np.asarray(prices[name])[count:]
        return 100 * np.asarray(implied_volatility(price, grid_t, grid_k, calls))

    smile = [
        ("starting point L0", "backbone", "backbone"),
        ("particle method", "particle", "particle"),
        *(
            (
                model_name(d, SMILE_ITERATIONS),
                f"{tag(d)}, {SMILE_ITERATIONS} iterations",
                kind_of(d),
            )
            for d in DEGREES
        ),
    ]
    phases = marker_phases(smile)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(WIDTH, 3.4),
        layout="constrained",
        sharex="col",
        gridspec_kw=dict(height_ratios=(2.2, 1.0)),
    )
    for column, (day, panel) in enumerate(days):
        above, below = axes[0, column], axes[1, column]
        rows = np.flatnonzero(grid_days == day)
        k = grid_k[rows]
        largest = []
        for name, _, kind in smile:
            values = iv(name)[rows]
            above.plot(k, values, zorder=3, **style(kind, phases.get(name)))
            if kind != "backbone":
                residual = 100 * (values - target[rows])
                below.plot(k, residual, zorder=3, **style(kind, phases.get(name)))
                largest.append(np.max(np.abs(residual)))
        above.plot(k, target[rows], color=COLORS["target"], linewidth=1.1, zorder=6)
        below.axhline(0.0, color=COLORS["target"], linewidth=0.9, zorder=5)
        limit = 1.15 * max(largest)
        below.set_ylim(-limit, limit)
        above.set_title(rf"{panel}: $T = {day}$ days", loc="left")
        below.set(xlabel=r"$k = \log(K/F)$")
        above.grid(**GRID)
        below.grid(**GRID)
    axes[0, 0].set_ylabel("implied volatility (%)")
    axes[1, 0].set_ylabel("IV minus target (bp)")
    handles = [
        Line2D([], [], color=COLORS["target"], linewidth=1.1, label="refined SVI target"),
    ] + [Line2D([], [], label=label, **style(kind)) for _, label, kind in smile]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "smiles")

    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.3), layout="constrained")
    for degrees in DEGREES:
        rms = np.asarray(increments[tuple(degrees)])
        left.plot(np.arange(1, rms.size + 1), rms, label=tag(degrees), **style(kind_of(degrees)))
    left.set(xlabel="fixed-point iteration $i$", ylabel=r"$\Delta_i$", yscale="log")
    left.set_xticks(range(1, ITERATIONS + 1))
    left.legend(title="degrees", frameon=False, loc="upper right")
    middle = (len(CORRECTION_STARTS) - 1) / 2
    dodge = {start: START_DODGE * (i - middle) for i, start in enumerate(CORRECTION_STARTS)}
    curves = {}
    for start in sorted(CORRECTION_STARTS, key=lambda value: value == SMILE_ITERATIONS):
        first = "starting point L0" if start == 0 else model_name(FIGURE_DEGREES, start)
        names = [first] + [
            model_name(FIGURE_DEGREES, start, step)
            for step in range(1, len(CORRECTION_RADII) + 1)
            if model_name(FIGURE_DEGREES, start, step) in statistics
        ]
        estimate = np.asarray([statistics[name][0][3] for name in names])
        error = np.asarray([statistics[name][1][3] for name in names])
        curves[start] = right.errorbar(
            np.arange(len(names)) + dodge[start],
            estimate,
            yerr=2 * error,
            capsize=2,
            **style("corrected", color=COLORS["starts"][start]),
        )
    right.set(xlabel="corrections", ylabel="bid-ask loss (bp)", yscale="log")
    right.set_xticks(range(len(CORRECTION_RADII) + 1))
    right.legend(
        [curves[start] for start in CORRECTION_STARTS],
        [f"from {start} iterations" for start in CORRECTION_STARTS],
        frameon=False,
        loc="center right",
    )
    for ax in (left, right):
        ax.grid(**GRID)
    save(fig, "convergence")

    spread = [
        ("particle method", "Particle, Guyon and Henry-Labordere", "particle"),
        (
            model_name(FIGURE_DEGREES, ITERATIONS),
            f"{tag(FIGURE_DEGREES)}, {ITERATIONS} iterations",
            "fixed",
        ),
        (
            corrected(prices, FIGURE_DEGREES, SMILE_ITERATIONS),
            f"{tag(FIGURE_DEGREES)}, {SMILE_ITERATIONS} iterations + "
            f"{len(CORRECTION_RADII)} corrections",
            "corrected",
        ),
    ]
    phases = marker_phases(spread)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.3), layout="constrained")
    for ax, (day, panel) in zip(axes, days, strict=True):
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
            for sign in (-1, 1):
                ax.plot(
                    quote_k[inner],
                    sign * band[inner],
                    color=COLORS["band_edge"],
                    linewidth=0.7,
                    zorder=2,
                )
        ax.axhline(0, color="#4B5563", linewidth=0.5, zorder=2)
        for name, _, kind in spread:
            values = (np.asarray(prices[name])[:count] - (bid + ask) / 2) * scale
            ax.plot(
                quote_k[rows],
                values[rows],
                color=COLORS[kind],
                linewidth=0.8,
                alpha=0.35,
                zorder=3,
                linestyle=KIND_STYLES[kind]["linestyle"],
            )
            if inner.size:
                ax.plot(quote_k[inner], values[inner], zorder=4, **style(kind, phases[name]))
        limit = 4 * np.percentile(band[inner] if inner.size else band[rows], 90)
        ax.set_ylim(-limit, limit)
        ax.set_title(rf"{panel}: $T = {day}$ days", loc="left")
        ax.set(xlabel=r"$k = \log(K/F)$")
        ax.grid(**GRID)
    axes[0].set_ylabel(r"$(P - P_{\mathrm{mid}})\,/\,\mathcal{V}$ (bp)")
    handles = [
        Patch(
            facecolor=COLORS["band_liquid"],
            edgecolor=COLORS["band_edge"],
            linewidth=0.7,
            label=r"bid-ask band, liquid quotes",
        ),
        Patch(facecolor=COLORS["band"], linewidth=0, label=r"bid-ask band, other quotes"),
    ] + [Line2D([], [], label=label, **style(kind)) for _, label, kind in spread]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    save(fig, "spreads")
