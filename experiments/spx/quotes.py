"""The SPX quotes of one day (Section 3.1 of the paper).

read_quotes parses a file written by pull_data.py and applies the rules of the paper:

- forward F(T) and discount factor D(T) of every expiry from put-call parity, by least squares on
  the mid prices of the strikes quoted on both sides;
- out-of-the-money quotes, puts for k = log(K/F) < 0 and calls otherwise, with prices divided by
  D F; puts are converted to calls by C = P + 1 - e^k;
- a positive bid, a relative spread (ask - bid)/(ask + bid) < 0.1, |k| < 0.2 + 0.4 T and
  0.02 < T <= 1.1;
- per calendar month the expiry with the most such quotes among those beyond T = 0.05, every
  expiry with T <= 0.05 and at least 150 quotes, and per expiry the more traded of the AM-settled
  SPX and the PM-settled SPXW series;
- the implied volatility sigma of the mid price, the Black vega and the Black delta, and the liquid
  quotes: traded on the day with absolute delta at least 0.1;
- the weights: every nonempty block of quotes of one maturity and option type has the same total
  weight, shared equally by its quotes; the liquid weights do the same over the liquid quotes.
"""

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from settings import (
    BAND,
    HORIZON,
    LIQUID_DELTA,
    LIQUID_VOLUME,
    MAXIMUM_RELATIVE_SPREAD,
    MINIMUM_MATURITY,
    MONTHLY_FROM,
    SHORT_EXPIRY_QUOTES,
)

from locsvol.pricing import black_vega, implied_volatility
from locsvol.targets import put_call_parity

COLUMNS = (
    "date",
    "exdate",
    "cp_flag",
    "strike",
    "best_bid",
    "best_offer",
    "impl_volatility",
    "delta",
    "volume",
    "open_interest",
    "root",
    "symbol",
    "optionid",
    "am_settlement",
    "ss_flag",
    "expiry_indicator",
    "forward_price",
)


@dataclass(frozen=True)
class Quotes:
    """Selected quotes in order of maturity and strike; prices are call prices divided by D F."""

    maturities: jax.Array
    log_strikes: jax.Array
    forwards: jax.Array
    discounts: jax.Array
    bid: jax.Array
    ask: jax.Array
    mid: jax.Array
    is_put: jax.Array
    implied_volatility: jax.Array
    vega: jax.Array
    delta: jax.Array
    volume: jax.Array
    liquid: jax.Array
    weights: jax.Array
    liquid_weights: jax.Array


def block_weights(maturities, puts, mask=None):
    """Equal total weight per nonempty (maturity, option type) block, shared equally by its
    quotes; with mask, over the quotes in mask, and zero elsewhere."""
    keep = np.ones(len(maturities), dtype=bool) if mask is None else np.asarray(mask)
    blocks = [(t, p) for t, p, k in zip(maturities, puts, keep, strict=True) if k]
    counts = Counter(blocks)
    weights = np.zeros(len(maturities))
    for i, (t, p) in enumerate(zip(maturities, puts, strict=True)):
        if keep[i]:
            weights[i] = 1 / (len(counts) * counts[(t, p)])
    return jnp.asarray(weights)


def contracts(path, day):
    """Both sides of every (expiry, series, strike) of the file, by expiry, series and strike."""
    with open(path, newline="") as stream:
        rows = list(csv.reader(stream))
    if tuple(rows[0]) != COLUMNS:
        raise ValueError(f"{path}: unexpected columns")
    column = {name: i for i, name in enumerate(COLUMNS)}
    found = {}
    for row in rows[1:]:
        if not row:
            continue
        if date.fromisoformat(row[column["date"]]) != day:
            raise ValueError(f"{path}: a quote of another day")
        series = row[column["root"]] or row[column["symbol"]].split(" ")[0]
        key = (date.fromisoformat(row[column["exdate"]]), series, float(row[column["strike"]]))
        found.setdefault(key, {})[row[column["cp_flag"]]] = dict(
            bid=float(row[column["best_bid"]]),
            ask=float(row[column["best_offer"]]),
            volume=float(row[column["volume"]] or 0),
        )
    missing = dict(bid=float("nan"), ask=float("nan"), volume=0.0)
    paired = []
    for line, ((expiry, series, strike), sides) in enumerate(sorted(found.items()), start=1):
        if expiry <= day:
            continue
        call, put = sides.get("C", missing), sides.get("P", missing)
        paired.append(
            dict(
                line=line,
                expiry=expiry.isoformat(),
                maturity=(expiry - day).days / 365,
                series=series,
                strike=strike,
                call_bid=call["bid"],
                call_ask=call["ask"],
                put_bid=put["bid"],
                put_ask=put["ask"],
                call_volume=call["volume"],
                put_volume=put["volume"],
            )
        )
    return paired


def carry(rows):
    """(forward, discount) of every expiry from put-call parity on the mid prices."""
    groups = defaultdict(list)
    for row in rows:
        prices = [row[n] for n in ("strike", "call_bid", "call_ask", "put_bid", "put_ask")]
        if (
            MINIMUM_MATURITY < row["maturity"] <= HORIZON
            and np.isfinite(prices).all()
            and row["strike"] > 0
            and 0 <= row["call_bid"] <= row["call_ask"]
            and 0 <= row["put_bid"] <= row["put_ask"]
        ):
            groups[row["expiry"]].append(row)
    expiries = sorted(groups)
    width = max(map(len, groups.values()))
    strikes, calls, puts, weights = (np.zeros((len(expiries), width)) for _ in range(4))
    strikes.fill(1.0)
    for i, expiry in enumerate(expiries):
        for j, row in enumerate(groups[expiry]):
            strikes[i, j] = row["strike"]
            calls[i, j] = (row["call_bid"] + row["call_ask"]) / 2
            puts[i, j] = (row["put_bid"] + row["put_ask"]) / 2
            weights[i, j] = 1
    discount, forward = jax.vmap(put_call_parity)(strikes, calls, puts, weights)
    return {e: (float(forward[i]), float(discount[i])) for i, e in enumerate(expiries)}


def more_traded_series(rows):
    """The series of an expiry with the largest summed volume of its quotes."""
    volume, count = defaultdict(float), Counter()
    for row in rows:
        volume[row["series"]] += row["volume"]
        count[row["series"]] += 1
    return max(volume, key=lambda series: (volume[series], count[series], series))


def read_quotes(path, day):
    """The selected quotes of the day in the file path; see the module docstring."""
    rows = contracts(path, day)
    forwards = carry(rows)
    kept = defaultdict(list)
    for row in rows:
        if row["expiry"] not in forwards:
            continue
        forward, discount = forwards[row["expiry"]]
        k = np.log(row["strike"] / forward)
        side = "call" if k >= 0 else "put"
        bid, ask = row[side + "_bid"], row[side + "_ask"]
        if (
            np.isfinite((bid, ask)).all()
            and 0 < bid <= ask
            and (ask - bid) / (ask + bid) < MAXIMUM_RELATIVE_SPREAD
            and abs(k) < BAND[0] + BAND[1] * row["maturity"]
        ):
            kept[row["expiry"]].append(
                dict(
                    row,
                    forward=forward,
                    discount=discount,
                    k=float(k),
                    put=side == "put",
                    bid=bid,
                    ask=ask,
                    volume=row[side + "_volume"],
                )
            )
    months = defaultdict(list)
    for expiry in kept:
        months[expiry[:7]].append(expiry)
    monthly = set()
    for candidates in months.values():
        later = [e for e in candidates if kept[e][0]["maturity"] > MONTHLY_FROM]
        if later:
            monthly.add(min(later, key=lambda e: (-len(kept[e]), e)))
    short = {
        e
        for e in kept
        if kept[e][0]["maturity"] <= MONTHLY_FROM and len(kept[e]) >= SHORT_EXPIRY_QUOTES
    }
    expiries = sorted(monthly | short)
    series = {e: more_traded_series(kept[e]) for e in expiries}
    selected = [row for e in expiries for row in kept[e] if row["series"] == series[e]]
    selected.sort(key=lambda row: (row["maturity"], row["strike"], row["line"]))
    t, k, f, d, bid, ask, volume = (
        jnp.asarray([row[name] for row in selected])
        for name in ("maturity", "k", "forward", "discount", "bid", "ask", "volume")
    )
    puts = jnp.asarray([row["put"] for row in selected])
    bid, ask = bid / (f * d), ask / (f * d)
    sigma = implied_volatility((bid + ask) / 2, t, k, ~puts)
    if not bool(jnp.all(jnp.isfinite(sigma))):
        raise ValueError("no implied volatility for some mid price")
    vega = black_vega(sigma, t, k)
    d1 = -k / (sigma * jnp.sqrt(t)) + sigma * jnp.sqrt(t) / 2
    delta = jnp.where(puts, norm.cdf(-d1), norm.cdf(d1))
    liquid = (volume >= LIQUID_VOLUME) & (delta >= LIQUID_DELTA)
    conversion = jnp.where(puts, 1.0 - jnp.exp(k), 0.0)
    maturities, is_put = [row["maturity"] for row in selected], [row["put"] for row in selected]
    return Quotes(
        maturities=t,
        log_strikes=k,
        forwards=f,
        discounts=d,
        bid=bid + conversion,
        ask=ask + conversion,
        mid=(bid + ask) / 2 + conversion,
        is_put=puts,
        implied_volatility=sigma,
        vega=vega,
        delta=delta,
        volume=volume,
        liquid=liquid,
        weights=block_weights(maturities, is_put),
        liquid_weights=block_weights(maturities, is_put, np.asarray(liquid)),
    )
