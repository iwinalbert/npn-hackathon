"""
Model card and the capability matrix.

THE CAPABILITY MATRIX IS A PRODUCT FEATURE, NOT A DISCLAIMER
------------------------------------------------------------
The research phase ran 86 experiments and rejected most of the techniques Use
Case 11's wording might tempt a team to claim: hierarchical reconciliation,
Croston/SBA/TSB, promotion modelling, segmentation. Hiding that and implying the
product does everything would be both dishonest and, on inspection, obviously
thin.

Shipping the rejections *with their measured numbers* is stronger: it shows the
use case was engaged with scientifically. Every entry below traces to a recorded
experiment, and nothing is asserted that the evidence does not support.
"""

from __future__ import annotations

from ..cache import ttl_cache
from ..db import backtest_source, query, query_one

_INT_FIELDS = {"n_estimators", "seed", "validation_n", "horizon_days", "n_series"}
_FLOAT_FIELDS = {"blend_weight_direct", "blend_weight_recursive",
                 "validation_rmse", "validation_mae"}


@ttl_cache()
def model_card() -> dict:
    rows = query("SELECT key, value FROM model_card")
    card = {r["key"]: r["value"] for r in rows}
    for k in _INT_FIELDS:
        if k in card:
            card[k] = int(card[k])
    for k in _FLOAT_FIELDS:
        if k in card:
            card[k] = float(card[k])
    return card


@ttl_cache()
def capabilities() -> dict:
    """
    What this system genuinely does, what research rejected, and what it cannot do.

    Categories:
      implemented   — built, and backed by validated work
      rejected      — investigated during research and turned down on evidence
      not_supported — outside what the frozen model or the data can support
    """
    implemented = [
        {"name": "28-day-ahead forecasts per store-item",
         "category": "implemented",
         "detail": ("All 30,490 store-item series forecast 28 days ahead from a "
                    "fixed origin (d_1941) by the frozen blend."),
         "evidence": "RMSE 2.0929 / MAE 1.0395 on 853,720 held-out predictions"},
        {"name": "Hierarchical aggregation",
         "category": "implemented",
         "detail": ("Navigate 12 aggregation levels. Every aggregate is an exact "
                    "sum of bottom-level forecasts, so the hierarchy is coherent "
                    "by construction."),
         "evidence": "Accuracy measured at all levels (Stage 7 audit)"},
        {"name": "Level-matched accuracy reporting",
         "category": "implemented",
         "detail": ("Accuracy is reported for the level you are viewing, because "
                    "the same forecast is ~28% accurate per store-item and ~95% "
                    "chain-wide."),
         "evidence": "uc11_hierarchy_levels.csv"},
        {"name": "Intermittent demand handling",
         "category": "implemented",
         "detail": ("Tweedie objective (variance_power 1.1) across all series; "
                    "every series labelled with its Syntetos-Boylan regime."),
         "evidence": ("Champion beats Croston, SBA, TSB and rolling-mean in "
                      "every one of the four regimes")},
        {"name": "Calendar, event and SNAP covariates",
         "category": "implemented",
         "detail": ("Holiday/event fields and state-matched SNAP flags are model "
                    "inputs and are shown alongside demand."),
         "evidence": "28/28 forecast days covered by calendar.csv"},
        {"name": "Price as a model covariate",
         "category": "implemented",
         "detail": ("sell_price and price-relative features are inputs to the "
                    "frozen model, and price is displayed with demand history."),
         "evidence": ("Ablation abl_4_plus_price improved RMSE; 30,490/30,490 "
                      "series priced for all 28 forecast days")},
        {"name": "Empirical backtest error bands",
         "category": "implemented",
         "detail": ("Observed p05-p95 of (actual - predicted) by demand regime "
                    "and horizon, variance-stabilised by sqrt(forecast). "
                    "Measured coverage 90.0%. Measured error, NOT a "
                    "model-produced prediction interval."),
         "evidence": "6.8M backtest rows across 8 disjoint origins"},
        {"name": "Historical validation replay",
         "category": "implemented",
         "detail": ("Predicted vs actual on 8 backtest windows where ground truth "
                    "exists, including the direct/recursive member split."),
         "evidence": "predictions/uc11_cache/ (champion reproductions)"},
        {"name": "Live model inference",
         "category": "implemented",
         "detail": ("The frozen boosters can be loaded and re-run on demand to "
                    "reproduce the shipped forecast, verifying the artefact end "
                    "to end."),
         "evidence": "Measured ~33 s for all 30,490 series"},
    ]

    rejected = [
        {"name": "Hierarchical reconciliation (top-down / MinT / middle-out)",
         "category": "rejected",
         "detail": ("Tested at three levels under two protocols. Above item "
                    "level there is almost nothing to recover; at item level the "
                    "method failed its pre-registered mechanism criterion."),
         "evidence": ("True-aggregate oracle ≤ -0.0221 above item level; "
                      "Experiments #80-#82, mechanism 2/4 windows")},
        {"name": "Croston / SBA / TSB",
         "category": "rejected",
         "detail": ("The classical intermittent-demand family was measured "
                    "against the champion in every demand regime and lost in all "
                    "of them."),
         "evidence": ("Champion 2.0920 vs Croston 2.2380, SBA 2.2317, TSB "
                      "2.2084 overall")},
        {"name": "Promotion / discount modelling",
         "category": "rejected",
         "detail": ("M5 contains no promotion field. A discount proxy against "
                    "the 52-week regular price was audited and had no residual "
                    "signal left to explain."),
         "evidence": "Perfect per-discount-bin correction worth only -0.0002 RMSE"},
        {"name": "Demand-regime segmentation (separate models per segment)",
         "category": "rejected",
         "detail": "Segment-specific recalibration has negligible headroom.",
         "evidence": "Per-regime rescale oracle -0.0008 RMSE"},
        {"name": "Hurdle / two-stage occurrence models",
         "category": "rejected",
         "detail": "Tested twice; both variants scored worse than the champion.",
         "evidence": "2.1267 and 2.1241 vs champion baseline"},
    ]

    not_supported = [
        {"name": "Price what-if / elasticity simulation",
         "category": "not_supported",
         "detail": ("The frozen model is a forecaster that uses price as context, "
                    "not a causal price-response model. Measured response to "
                    "simulated price changes is non-monotone and sometimes "
                    "economically backwards (a 10% cut predicted -74% demand on "
                    "one high-volume series), so no such control is offered."),
         "evidence": "Architecture plan §14.3, measured on three series"},
        {"name": "Model-produced prediction intervals",
         "category": "not_supported",
         "detail": ("The model emits point forecasts only. The bands shown are "
                    "empirical backtest error, clearly labelled, not a predictive "
                    "distribution."),
         "evidence": None},
        {"name": "Accuracy for the delivered forecast window",
         "category": "not_supported",
         "detail": ("No ground truth exists for d_1942-d_1969, so no accuracy "
                    "figure is shown against it. All accuracy comes from held-out "
                    "windows where actuals exist."),
         "evidence": None},
        {"name": "Stockout / censored-demand detection",
         "category": "not_supported",
         "detail": ("A recorded zero may mean 'nobody wanted it' or 'it was not "
                    "on the shelf'. The dataset cannot distinguish them."),
         "evidence": None},
        {"name": "Live retraining",
         "category": "not_supported",
         "detail": ("The model is frozen. Retraining is a deliberate research "
                    "action with its own validation protocol, not a product "
                    "feature."),
         "evidence": "02_DOCUMENTATION/02_MODEL/MODEL_FREEZE.md"},
    ]

    return {"implemented": implemented, "rejected": rejected,
            "not_supported": not_supported}


@ttl_cache()
def provenance() -> dict:
    """Where every served number comes from."""
    card = model_card()
    counts = {}
    for t in ("series", "forecast", "calendar", "error_bands",
              "level_accuracy", "window_metrics"):
        row = query_one(f"SELECT count(*) AS n FROM {t}")
        counts[t] = int(row["n"]) if row else 0
    counts["backtest"] = int(
        query_one(f"SELECT count(*) AS n FROM {backtest_source()}")["n"])
    origins = query(
        f"SELECT DISTINCT origin_idx FROM {backtest_source()} "
        "ORDER BY origin_idx")
    return {
        "model_direct_sha256": card.get("model_direct_sha256"),
        "model_recursive_sha256": card.get("model_recursive_sha256"),
        "forecast_sha256": card.get("forecast_sha256"),
        "db_built_at": card.get("db_built_at"),
        "row_counts": counts,
        "backtest_origins": [f"d_{int(o['origin_idx']) + 1}" for o in origins],
        "sources": {
            "forecast": ("predictions/final_forecast/"
                         "final_forecast_28day_v3_diversity_blend.csv"),
            "history": ("01_PROJECT/backend/data/history.parquet — product-owned "
                        "sidecar built read-only from "
                        "data/processed/sales_long_full.parquet"),
            "backtest": ("01_PROJECT/backend/data/backtest.parquet — product-owned "
                         "sidecar built read-only from predictions/uc11_cache/ "
                         "(8 champion reproductions)"),
            "level_accuracy": "experiments/artifacts/uc11_hierarchy_levels.csv",
            "model": ("models/champion/model_11_blend_direct_final_forecast.txt "
                      "+ model_12_blend_recursive_shape_final.txt"),
        },
    }
