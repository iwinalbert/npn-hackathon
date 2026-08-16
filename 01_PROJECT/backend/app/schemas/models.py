"""
Response schemas.

These are the API's contract. They are deliberately explicit about provenance —
several carry a `basis` or `disclaimer` field — because this product serves
forecasts from a frozen research model and the difference between "the model
predicted this" and "we measured this in backtest" must survive the trip to the
UI. A field name that blurs the two would eventually become a misleading chart.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------

class ModelCard(BaseModel):
    model_name: str
    blend_formula: str
    blend_weight_direct: float
    blend_weight_recursive: float
    objective: str
    n_estimators: int
    seed: int
    status: str = Field(description="FROZEN — the model must not be modified")
    validation_rmse: float
    validation_mae: float
    validation_window: str
    validation_n: int
    forecast_origin: str
    forecast_dates: str
    horizon_days: int
    n_series: int
    model_direct_sha256: str
    model_recursive_sha256: str
    forecast_sha256: str
    db_built_at: str


class Capability(BaseModel):
    name: str
    category: str = Field(
        description="implemented | rejected | not_supported"
    )
    detail: str
    evidence: str | None = None


class CapabilityMatrix(BaseModel):
    implemented: list[Capability]
    rejected: list[Capability]
    not_supported: list[Capability]


# --------------------------------------------------------------------------
# hierarchy
# --------------------------------------------------------------------------

class LevelInfo(BaseModel):
    level: str
    label: str
    node_count: int
    columns: list[str]


class HierarchyNode(BaseModel):
    level: str
    node_id: str
    label: str
    n_series: int
    mean_daily_sales: float | None = None


class SeriesSummary(BaseModel):
    series_idx: int
    id: str
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str
    volume_tier: str
    regime: str
    mean_daily_sales: float
    zero_pct: float


# --------------------------------------------------------------------------
# series detail
# --------------------------------------------------------------------------

class SeriesDetail(SeriesSummary):
    total_units: int
    adi: float
    cv2: float
    regime_explanation: str


class HistoryPoint(BaseModel):
    date: str
    day_idx: int
    sales: int
    sell_price: float | None = None
    event_name: str | None = None
    snap: int = 0


class ForecastPoint(BaseModel):
    date: str
    day_idx: int
    horizon: int
    yhat: float
    lower: float | None = Field(
        default=None,
        description="Empirical backtest error band, NOT a model interval")
    upper: float | None = None


class SeriesForecast(BaseModel):
    series: SeriesSummary
    origin_day: str
    origin_date: str
    forecast: list[ForecastPoint]
    total_28d: float
    band_basis: str = Field(
        default=("Empirical p05-p95 of (actual - predicted) observed on held-out "
                 "backtest windows for series of this volume tier at this "
                 "horizon. This is measured error, NOT a model-produced "
                 "prediction interval.")
    )


class SeriesHistoryResponse(BaseModel):
    series: SeriesSummary
    history: list[HistoryPoint]
    from_date: str
    to_date: str


# --------------------------------------------------------------------------
# aggregate forecast
# --------------------------------------------------------------------------

class AggregatePoint(BaseModel):
    date: str
    day_idx: int
    horizon: int
    yhat: float


class AggregateForecast(BaseModel):
    level: str
    node_id: str
    n_series: int
    origin_day: str
    forecast: list[AggregatePoint]
    total_28d: float
    coherence_note: str = Field(
        default=("Aggregates are exact sums of the bottom-level store-item "
                 "forecasts, so the hierarchy is coherent by construction.")
    )
    expected_accuracy: dict | None = Field(
        default=None,
        description="Measured accuracy at this aggregation level, if available")


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    app: str


class ReadinessResponse(BaseModel):
    ready: bool
    detail: dict
    cache: dict
