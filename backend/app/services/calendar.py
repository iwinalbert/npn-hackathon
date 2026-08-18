
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
    return f"d_{int(day_idx) + 1}"


def calendar_row(day_idx: int) -> dict:
    return _calendar()[int(day_idx)]


def snap_for_state(day_idx: int, state_id: str) -> int:
    row = _calendar().get(int(day_idx))
    if row is None:
        return 0
    return int(row.get(f"snap_{state_id}", 0) or 0)


def origin_day_idx() -> int:
    return settings.forecast_origin_idx


def forecast_days() -> list[int]:
    o = origin_day_idx()
    return [o + h for h in range(1, settings.horizon + 1)]
