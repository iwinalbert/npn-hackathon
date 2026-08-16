"""
DuckDB access layer.

THREE GUARANTEES THIS MODULE ENFORCES
-------------------------------------
1. **Read-only.** The product database is opened with ``read_only=True`` and the
   sidecar parquet files are only ever read. Nothing in the API can write to any
   artefact, product-owned or research-owned.

2. **No string-interpolated user input.** Every value is a bound parameter. The
   only identifiers ever interpolated are module constants (``SAFE_LEVELS``) and
   artefact paths from settings — never request data.

3. **No baked absolute paths.** Sidecar locations are resolved from settings on
   every call, so the same database file works on a laptop and in a container
   with the data mounted anywhere. (DuckDB bakes absolute paths into view SQL,
   which is why this API defines no views over external files.)

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
from .errors import ServiceUnavailable

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

SAFE_SERIES_COLUMNS: frozenset[str] = frozenset(
    {"series_idx", "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
     "mean_daily_sales", "mean_daily_28d", "mean_daily_91d", "total_units",
     "volume_tier", "regime", "adi", "cv2", "zero_pct"}
)

TABLES = ("series", "forecast", "calendar", "error_bands", "level_accuracy",
          "window_metrics", "model_card")

_lock = threading.Lock()
_conn: duckdb.DuckDBPyConnection | None = None


class DatabaseUnavailable(ServiceUnavailable):
    """The product data layer has not been built, or is not readable."""

    def __init__(self, message: str, **ctx):
        super().__init__(message, remedy="python tasks.py build-db", **ctx)


def history_source() -> str:
    """SQL fragment for the history sidecar, resolved fresh from settings."""
    p = Path(settings.history_parquet)
    if not p.exists():
        raise DatabaseUnavailable(f"history sidecar not found at {p}")
    return f"read_parquet('{p.as_posix()}')"


def backtest_source() -> str:
    """SQL fragment for the backtest sidecar, resolved fresh from settings."""
    p = Path(settings.backtest_parquet)
    if not p.exists():
        raise DatabaseUnavailable(f"backtest sidecar not found at {p}")
    return f"read_parquet('{p.as_posix()}')"


def get_connection() -> duckdb.DuckDBPyConnection:
    """The shared read-only connection, opened lazily on first use."""
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                db = Path(settings.product_db)
                if not db.exists():
                    raise DatabaseUnavailable(
                        f"product database not found at {db}")
                try:
                    _conn = duckdb.connect(str(db), read_only=True)
                except duckdb.Error as exc:                     # pragma: no cover
                    raise DatabaseUnavailable(
                        f"could not open {db}: {exc}") from exc
    return _conn


def close_connection() -> None:
    global _conn
    with _lock:
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
            f"unknown level '{level}'. Valid levels: {sorted(SAFE_LEVELS)}")
    return SAFE_LEVELS[level]


def health() -> dict[str, Any]:
    """
    Readiness detail: which artefacts exist, which are queryable, row counts.

    Never raises — readiness must be able to report a broken state rather than
    become one.
    """
    out: dict[str, Any] = {
        "data_dir": str(settings.data_dir),
        "artefacts": {},
        "tables": {},
        "history_queryable": False,
        "backtest_queryable": False,
        "errors": [],
    }
    for name, path in settings.required_artefacts.items():
        p = Path(path)
        out["artefacts"][name] = {
            "path": str(p),
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None,
        }

    if not Path(settings.product_db).exists():
        out["errors"].append("product.duckdb missing")
        return out

    try:
        for t in TABLES:
            row = query_one(f"SELECT count(*) AS n FROM {t}")
            out["tables"][t] = int(row["n"]) if row else 0
    except Exception as exc:                                   # noqa: BLE001
        out["errors"].append(f"product_db: {type(exc).__name__}: {exc}")
        return out

    for label, fn in (("history_queryable", history_source),
                      ("backtest_queryable", backtest_source)):
        try:
            query_one(f"SELECT 1 AS ok FROM {fn()} LIMIT 1")
            out[label] = True
        except Exception as exc:                               # noqa: BLE001
            out["errors"].append(f"{label}: {type(exc).__name__}: {exc}")
    return out
