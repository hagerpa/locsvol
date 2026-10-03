"""Download the SPX option days of the study from WRDS (OptionMetrics IvyDB US).

    python experiments/spx/pull_data.py --username <wrds user> [--output data/wrds]

writes spx_optionmetrics_<date>.csv for both days of every pair, with the contracts of the day
ordered by expiry, root, strike and side, strikes in index points, and checks each file against
the SHA-256 in date_pairs.py. Requires a WRDS account with access to OptionMetrics; the data are
licensed and must not be redistributed.
"""

import argparse
from datetime import date
from hashlib import sha256
from pathlib import Path

from date_pairs import PAIRS

SECID = 108105  # the S&P 500 index in IvyDB US
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


def query(year):
    """All SPX contracts of one day in the option table of its year."""
    names = ", ".join(
        "strike_price / 1000.0 as strike" if name == "strike" else name for name in COLUMNS
    )
    return (
        f"select {names} from optionm.opprcd{year} where secid = {SECID} and date = %(day)s "
        "order by exdate, root, strike_price, cp_flag, optionid"
    )


def download(connection, day: date, path: Path):
    """Write the option file of one day and return its SHA-256."""
    options = connection.raw_sql(query(day.year), params={"day": day.isoformat()})
    if options.empty:
        raise RuntimeError(f"no SPX options on {day}")
    frame = options.loc[:, list(COLUMNS)].sort_values(
        ["exdate", "root", "strike", "cp_flag", "optionid"], kind="stable"
    )
    path.write_text(frame.to_csv(index=False, lineterminator="\n"))
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", help="WRDS username; otherwise the client asks")
    parser.add_argument("--output", type=Path, default=Path("data/wrds"))
    arguments = parser.parse_args()
    import wrds

    connection = wrds.Connection(
        **({"wrds_username": arguments.username} if arguments.username else {})
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    try:
        for pair in PAIRS.values():
            for day, expected in (
                (pair.earlier, pair.earlier_sha256),
                (pair.event_day, pair.event_sha256),
            ):
                path = arguments.output / f"spx_optionmetrics_{day.isoformat()}.csv"
                digest = download(connection, day, path)
                status = "ok" if digest == expected else "differs from date_pairs.py"
                print(f"{path.name}: {status}", flush=True)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
