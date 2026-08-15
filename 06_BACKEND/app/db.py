"""
DuckDB access layer.

TWO GUARANTEES THIS MODULE ENFORCES
-----------------------------------
1. **Read-only.** The product database is opened with ``read_only=True`` and the
   research parquet is only ever read through ``read_parquet``. Nothing in the
   API can write to the research tree even if a caller tried.

2. **No string-interpolated user input.** Every value is passed as a bound
   parameter. The only identifiers ever interpolated are drawn from
   ``SAFE_COLUMNS`` / ``SAFE_LEVELS``, which are module constants, never request
   data. This is the single most important defence for an analytical API whose
   whole job is turning URLs into SQL.

CONCURRENCY
-----------
A DuckDB connection is not safe to share across threads, but ``con.cursor()``
returns an independent handle onto the same database. FastAPI runs sync route
handlers in a threadpool, so each request takes its own cursor from one shared
read-only connection.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import duckdb

from .config import settings

# --- identifier whitelists --------------------------------------------------
# Anything a request can influence must appear here, or it cannot reach SQL.

SAFE_LEVELS: dict[str, list[str]] = {
    "total": [],
    "state": ["state_id"],
    "store": ["store_id"],
    "category": ["cat_id"],
    "department": ["dept_id"],
    "state_category": ["state_id", "cat_id"],
    "state_department": ["state_id", "dept_id"],
    "store_category": ["store_id", "cat_id"],
    "store_department": ["store_id", "dept_id"],
    "item": ["item_id"],
    "item_state": ["item_id", "state_id"],
    "series": ["store_id", "item_id"],
}

SAFE_COLUMNS: frozenset[str] = frozenset(
    {"series_idx", "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
     "mean_daily_sales", "total_units", "volume_tier", "regime", "adi", "cv2",
     "zero_pct"}
)

_lock = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None


class DatabaseUnavailable(RuntimeError):
    """Raised when the product database has not been built yet."""


def _panel_source() -> str:
    """
    The history panel, referenced by path rather than through a baked view.

    The build script also creates a ``panel`` view, but that view stores an
    absolute path from build time. Referencing the parquet through settings
    instead means the same database file works when the research tree is mounted
    at a different location inside a container.
    """
    return f"read_parquet('{Path(settings.panel_parquet).as_posix()}')"


PANEL = _panel_source()


def get_connection() -> duckdb.DuckDBPyConnection:
    """The shared read-only connection, opened lazily on first use."""
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                db = Path(settings.product_db)
                if not db.exists():
                    raise DatabaseUnavailable(
                        f"product database not found at {db}. "
                        "Build it with: python 06_BACKEND/scripts/build_product_db.py"
                    )
                _conn = duckdb.connect(str(db), read_only=True)
    return _conn


def close_connection() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def query(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    """Run a parameterised query and return a list of dicts."""
    cur = get_connection().cursor()
    try:
        rel = cur.execute(sql, params or [])
        cols = [d[0] for d in rel.description]
        return [dict(zip(cols, row)) for row in rel.fetchall()]
    finally:
        cur.close()


def query_one(sql: str, params: list[Any] | None = None) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def validate_level(level: str) -> list[str]:
    """Map a level name to its grouping columns, or raise."""
    if level not in SAFE_LEVELS:
        raise ValueError(
            f"unknown level '{level}'. Valid levels: {sorted(SAFE_LEVELS)}"
        )
    return SAFE_LEVELS[level]


def health() -> dict[str, Any]:
    """Readiness detail: what exists, what is queryable, how big."""
    out: dict[str, Any] = {
        "product_db_path": str(settings.product_db),
        "product_db_exists": Path(settings.product_db).exists(),
        "panel_parquet_exists": Path(settings.panel_parquet).exists(),
        "tables": {},
        "panel_queryable": False,
    }
    if not out["product_db_exists"]:
        return out
    try:
        for t in ("series", "forecast", "backtest", "error_bands",
                  "level_accuracy", "model_card"):
            row = query_one(f"SELECT count(*) AS n FROM {t}")
            out["tables"][t] = row["n"] if row else 0
    except Exception as exc:                                   # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        query_one(f"SELECT 1 AS ok FROM {PANEL} LIMIT 1")
        out["panel_queryable"] = True
    except Exception as exc:                                   # noqa: BLE001
        out["panel_error"] = f"{type(exc).__name__}: {exc}"
    return out
