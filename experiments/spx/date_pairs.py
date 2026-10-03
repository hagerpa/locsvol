"""The pairs of days of the study.

The backbone is fitted on the earlier day, a trading day about four weeks before the event
without a scheduled macro release; the leverage factor is calibrated on the day of the event.
For each day, sha256 is the SHA-256 of its OptionMetrics file as pull_data.py writes it.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DatePair:
    label: str
    earlier: date
    event_day: date
    event: str
    earlier_sha256: str
    event_sha256: str


PAIRS = {
    pair.label: pair
    for pair in (
        DatePair(
            "svb",
            date(2023, 2, 13),
            date(2023, 3, 13),
            "failure of Silicon Valley Bank; front-end rates collapse, VIX 26",
            "d04364686f31f76a108b77d7a46108620f1bb490edf339ef0b62195cf30172a7",
            "7267bbae5fab6858172f2dd9b2ee555afb6a75dadd467146225dad6d8eb633dd",
        ),
        DatePair(
            "carry-unwind",
            date(2024, 7, 8),
            date(2024, 8, 5),
            "unwind of the yen carry trade; VIX intraday above 60, SPX about -3%",
            "46a02895c44126075bf897cca33fc1bba938a2b3cfcd92e3a34dc41f4a67a5fe",
            "65a6ca69123f6e14491e0824f7b648acf0c1858b4aab1b4fb16430c49199415e",
        ),
        DatePair(
            "election",
            date(2024, 10, 9),
            date(2024, 11, 6),
            "day after the US presidential election; event premium released, SPX about +2.5%",
            "584110f2faeb0b69354b63961b33331a09c8242bd39327c7ab4d207e083af397",
            "b977aaedf72bda00594a7eba3a7832f6a6f1af173f0272807e5e77385465948d",
        ),
        DatePair(
            "fomc",
            date(2024, 11, 20),
            date(2024, 12, 18),
            "interest rate decision of the Federal Open Market Committee; SPX about -3%, "
            "VIX from 15 to 28",
            "8d96b5db557f0cf3fa20603223e31405d1947bf0751000028904d1d41e9b3fee",
            "ab63c4a61891436e640837932a504d048d8e327c3d55a9648cc2a35fb154e1ab",
        ),
        DatePair(
            "tariffs",
            date(2025, 2, 19),
            date(2025, 4, 4),
            "second day after the tariff announcement of 2 April 2025; SPX about -6%, VIX 45; "
            "the earlier day is six weeks before, since March 2025 was itself agitated",
            "7a7e2b2c9add13979590bbb156c1e9c8cc8223adad9c37d8c951923fca7647bf",
            "e36463391f4d7a9709317d4a88bbfa9521e30deb60a765dfbfded9b781045827",
        ),
    )
}
