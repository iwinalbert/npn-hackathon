"""
Day-index <-> date translation.

The research pipeline works in zero-based day indices (idx 0 == d_1 ==
2011-01-29); humans and charts work in dates. The mapping is loaded once from
the `calendar` table in the product database — which was built from the same
calendar.csv the model itself was trained against — so the product cannot drift
from the research layer's notion of what day d_1942 is.

Reading it from the database rather than the CSV means the API needs no access
to data/raw/ at runtime, which is what lets the container run without the
research tree mounted.
"""

from __future__ import annotations

import threading

from ..config import settings
from ..db import query

_lock = threading.Lock()
_cache: dict[int, dict] | None = None


def _calendar() -> dict[int, dict]:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                rows = query(
                    "SELECT day_idx, date, wday, month, year, is_weekend, "
                    "       event_name_1, event_type_1, event_name_2, "
                    "       event_type_2, snap_CA, snap_TX, snap_WI "
                    "FROM calendar ORDER BY day_idx")
                _cache = {int(r["day_idx"]): r for r in rows}
    return _cache


def reset_cache() -> None:
    global _cache
    with _lock:
        _cache = None


def date_of(day_idx: int) -> str:
    row = _calendar().get(int(day_idx))
    if row is None:
        raise KeyError(f"day_idx {day_idx} is outside the calendar")
    return str(row["date"])[:10]


def day_label(day_idx: int) -> str:
    """Zero-based day index -> the research layer's 'd_N' label."""
    return f"d_{int(day_idx) + 1}"


def calendar_row(day_idx: int) -> dict:
    return _calendar()[int(day_idx)]


def snap_for_state(day_idx: int, state_id: str) -> int:
    row = _calendar().get(int(day_idx))
    if row is None:
        return 0
    return int(row.get(f"snap_{state_id}", 0) or 0)


def origin_day_idx() -> int:
    """Zero-based index of the forecast origin: 1940 == d_1941 == 2016-05-22."""
    return settings.forecast_origin_idx


def forecast_days() -> list[int]:
    """The 28 zero-based day indices the frozen forecast covers (1941..1968)."""
    o = origin_day_idx()
    return [o + h for h in range(1, settings.horizon + 1)]
