"""
Day-index ↔ date translation.

The research pipeline works in zero-based day indices (`idx 0 == d_1 ==
2011-01-29`); humans and charts work in dates. Rather than re-deriving the
mapping in five places, it is loaded once from the calendar file the model
itself was built against, so the product can never drift from the research
layer's notion of what day `d_1942` is.

`calendar.csv` covers d_1..d_1969 — 28 days past the end of the sales history —
which is exactly why the forecast window has known covariates.
"""

from __future__ import annotations

import csv
from functools import lru_cache

from ..config import settings


@lru_cache(maxsize=1)
def _calendar() -> dict[int, dict]:
    """day_idx -> {date, wday, month, year, event_name_1, snap_CA/TX/WI}."""
    out: dict[int, dict] = {}
    with open(settings.calendar_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            # "d_123" -> day_idx 122 (zero-based)
            idx = int(row["d"][2:]) - 1
            out[idx] = {
                "date": row["date"],
                "wday": int(row["wday"]),
                "month": int(row["month"]),
                "year": int(row["year"]),
                "event_name_1": row["event_name_1"] or None,
                "event_type_1": row["event_type_1"] or None,
                "snap_CA": int(row["snap_CA"]),
                "snap_TX": int(row["snap_TX"]),
                "snap_WI": int(row["snap_WI"]),
            }
    return out


def date_of(day_idx: int) -> str:
    """Zero-based day index -> ISO date. Raises KeyError past the calendar."""
    return _calendar()[day_idx]["date"]


def day_label(day_idx: int) -> str:
    """Zero-based day index -> the research layer's 'd_N' label."""
    return f"d_{day_idx + 1}"


def calendar_row(day_idx: int) -> dict:
    return _calendar()[day_idx]


def snap_for_state(day_idx: int, state_id: str) -> int:
    row = _calendar()[day_idx]
    return int(row.get(f"snap_{state_id}", 0))


def forecast_days() -> list[int]:
    """The 28 zero-based day indices the frozen forecast covers (1941..1968)."""
    return [settings.forecast_origin_idx + h
            for h in range(1, settings.horizon + 1)]


def origin_day_idx() -> int:
    """Zero-based index of the forecast origin: 1940 == d_1941 == 2016-05-22."""
    return settings.forecast_origin_idx
