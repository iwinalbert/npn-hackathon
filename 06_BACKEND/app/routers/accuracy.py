"""Model performance from verified backtest artefacts."""
from __future__ import annotations

from fastapi import APIRouter, Query

from ..services import accuracy as svc

router = APIRouter(prefix="/accuracy", tags=["accuracy"])

PRIMARY = 1912


@router.get("/windows",
            summary="Backtest windows where ground truth exists")
def windows() -> list[dict]:
    """
    Only windows with real observed sales are listed. The delivered forecast
    window (d_1942-d_1969) is deliberately absent: it has no ground truth, so no
    accuracy figure can honestly be quoted against it.
    """
    return svc.windows()


@router.get("/levels", summary="Measured accuracy at each aggregation level")
def levels() -> dict:
    return {
        "levels": svc.by_level(),
        "note": ("The SAME forecast is ~28% accurate per store-item and ~97% "
                 "chain-wide. Errors are largely independent and cancel when "
                 "summed. Always quote the level that matches the decision."),
    }


@router.get("/horizon", summary="How error grows across the 28-day horizon")
def horizon(origin_idx: int = Query(default=PRIMARY)) -> list[dict]:
    return svc.by_horizon(origin_idx)


@router.get("/regimes", summary="Accuracy per Syntetos-Boylan demand regime")
def regimes(origin_idx: int = Query(default=PRIMARY)) -> dict:
    return {
        "origin_idx": origin_idx,
        "regimes": svc.by_regime(origin_idx),
        "note": ("Regimes classify how intermittent a series is. This is the "
                 "direct evidence for the intermittent-demand requirement."),
    }


@router.get("/members", summary="Direct vs recursive vs blend decomposition")
def members(origin_idx: int = Query(default=PRIMARY)) -> dict:
    return svc.members(origin_idx)


@router.get("/error-bands", summary="The empirical error-band table")
def error_bands(regime: str | None = Query(default=None)) -> dict:
    return {
        "bands": svc.error_bands(regime),
        "basis": ("Quantiles of (actual - predicted) / sqrt(max(forecast, 1)) "
                  "measured on 8 held-out backtest windows. Reconstruct with: "
                  "lower = max(0, yhat + q05 * sqrt(max(yhat, 1)))."),
        "measured_coverage": 0.90,
        "disclaimer": ("Observed model error. NOT a model-produced prediction "
                       "interval — the frozen model emits point forecasts only."),
    }


@router.get("/backtest/{store_id}/{item_id}",
            summary="Predicted vs actual for one series")
def series_backtest(store_id: str, item_id: str,
                    origin_idx: int = Query(default=PRIMARY)) -> dict:
    return svc.series_backtest(store_id, item_id, origin_idx)


@router.get("/backtest", summary="Predicted vs actual for a hierarchy node")
def aggregate_backtest(
    level: str = Query(default="store"),
    node_id: str = Query(default="CA_1"),
    origin_idx: int = Query(default=PRIMARY),
) -> dict:
    return svc.aggregate_backtest(level, node_id, origin_idx)


@router.get("/occurrence",
            summary="Demand-occurrence diagnostics (did it spot the selling days?)")
def occurrence(origin_idx: int = Query(default=PRIMARY)) -> dict:
    """
    Classification-style diagnostics from thresholding the point forecast at
    0.5 units. The model was never trained to classify — these describe a
    behaviour, they are not the task metric.
    """
    return svc.occurrence(origin_idx)


@router.get("/volume-tiers", summary="Accuracy by demand volume")
def volume_tiers(origin_idx: int = Query(default=PRIMARY)) -> dict:
    return svc.by_volume_tier(origin_idx)
