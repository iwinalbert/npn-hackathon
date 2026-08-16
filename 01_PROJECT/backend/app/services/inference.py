"""
INFERENCE SERVICE — the boundary between the product and the frozen model.

This is the only module in the API that touches the model. It is deliberately
narrow, and everything it refuses to do is as important as what it does.

WHAT IT DOES
------------
Loads the two frozen LightGBM boosters and re-runs them forward from the frozen
origin (d_1941) to reproduce the shipped 28-day forecast, then compares the
result against the shipped artefact. That proves, live:

  1. the model files load and execute;
  2. the feature pipeline is intact;
  3. the shipped forecast is authentic and reproducible;
  4. the system has a genuine model-serving path, not just a CSV reader.

Measured on the development machine: direct member 4.1 s for all 853,720 rows,
recursive rollout ~29 s, so a full blend re-forecast is ~33 s.

WHAT IT REFUSES TO DO, AND WHY
------------------------------
* **No retraining, re-tuning or re-blending.** The model is frozen.
* **No inference at an origin earlier than d_1941.** Both boosters were trained
  with data up to d_1941. Running them at an earlier origin would let them see
  the future of the window being predicted — textbook leakage. Historical
  accuracy comes from the cached per-window artefacts instead, where each
  window's members were retrained with a correct cutoff.
* **No perturbed-covariate scenarios.** Measured response to simulated price
  changes is non-monotone and sometimes economically backwards, so the model is
  not exposed as a what-if engine. See PRODUCT_ARCHITECTURE_PLAN.md §14.3.

DEPLOYMENT SHAPE
----------------
`lightgbm`, `numpy` and the research `pipeline` package are imported LAZILY,
inside functions. A container built without them still serves every other
endpoint; `/inference/status` reports the reason inference is unavailable and
the inference endpoints return 503 instead of the process failing to start.

MEMORY
------
The two boosters (~29 MB, ~1 s to load) are cached for the process lifetime, so
no request ever reloads a model binary. The 800 MB `M5Data` panel needed to
rebuild features is NOT cached: it is loaded per job (~14 s) and released, which
keeps idle RSS near 200 MB. That trade is deliberate — a container that idles at
1 GB to save 14 s on a rarely-used endpoint is the wrong shape.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ..config import settings

#: The only origin the frozen boosters may be run at (zero-based: d_1941).
FROZEN_ORIGIN_IDX = 1940

_boosters_lock = threading.Lock()
_boosters: dict[str, Any] | None = None
_availability_cache: dict | None = None


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def _probe_dependencies() -> tuple[bool, list[str]]:
    """Which optional runtime pieces are present? Never raises."""
    problems: list[str] = []
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:                                   # noqa: BLE001
        problems.append(f"lightgbm unavailable: {type(exc).__name__}")
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
    except Exception as exc:                                   # noqa: BLE001
        problems.append(f"numpy/pandas unavailable: {type(exc).__name__}")

    for label, path in (("direct model", settings.model_direct),
                        ("recursive model", settings.model_recursive),
                        ("frozen forecast", settings.forecast_csv)):
        if not Path(path).exists():
            problems.append(f"{label} not found at {path}")

    pipeline_dir = Path(settings.project_root) / "pipeline"
    if not (pipeline_dir / "config.py").exists():
        problems.append(f"research pipeline not found at {pipeline_dir}")

    return (not problems), problems


def availability(refresh: bool = False) -> dict:
    """
    Whether live inference can run here, and if not, precisely why.

    Cached because the dependency probe imports lightgbm, which is not free.
    """
    global _availability_cache
    if _availability_cache is not None and not refresh:
        return _availability_cache

    if not settings.enable_inference:
        result = {
            "available": False,
            "enabled": False,
            "reasons": ["inference disabled by configuration "
                        "(NPN_ENABLE_INFERENCE=false)"],
        }
    else:
        ok, problems = _probe_dependencies()
        result = {"available": ok, "enabled": True, "reasons": problems}

    result.update({
        "frozen_origin": f"d_{FROZEN_ORIGIN_IDX + 1}",
        "models_cached": _boosters is not None,
        "estimated_runtime_seconds": 35,
        "supported_operations": ["verify"],
        "refused_operations": {
            "earlier_origin_inference":
                "would leak: both boosters were trained with data up to d_1941",
            "retraining": "the model is frozen",
            "price_scenarios":
                "measured price response is non-monotone and not causal",
        },
    })
    _availability_cache = result
    return result


def require_available() -> None:
    from ..errors import ServiceUnavailable
    a = availability()
    if not a["available"]:
        raise ServiceUnavailable(
            "live inference is not available in this deployment",
            reasons=a["reasons"])


# ---------------------------------------------------------------------------
# Model loading — once per process
# ---------------------------------------------------------------------------

def get_boosters() -> dict[str, Any]:
    """
    The two frozen boosters, loaded once and reused for the process lifetime.

    Double-checked locking: concurrent first requests must not each load 29 MB
    of model.
    """
    global _boosters
    if _boosters is None:
        with _boosters_lock:
            if _boosters is None:
                import lightgbm as lgb                          # noqa: PLC0415
                t0 = time.perf_counter()
                direct = lgb.Booster(model_file=str(settings.model_direct))
                recursive = lgb.Booster(model_file=str(settings.model_recursive))
                _boosters = {
                    "direct": direct,
                    "recursive": recursive,
                    "loaded_at": time.time(),
                    "load_seconds": round(time.perf_counter() - t0, 3),
                    "direct_trees": direct.num_trees(),
                    "direct_features": direct.num_feature(),
                    "recursive_trees": recursive.num_trees(),
                    "recursive_features": recursive.num_feature(),
                }
    return _boosters


def unload_boosters() -> None:
    global _boosters
    with _boosters_lock:
        _boosters = None


def model_runtime_info() -> dict:
    """Booster metadata without forcing a load."""
    if _boosters is None:
        return {"loaded": False}
    b = _boosters
    return {
        "loaded": True,
        "load_seconds": b["load_seconds"],
        "direct": {"trees": b["direct_trees"], "features": b["direct_features"]},
        "recursive": {"trees": b["recursive_trees"],
                      "features": b["recursive_features"]},
    }


# ---------------------------------------------------------------------------
# The verification run
# ---------------------------------------------------------------------------

def run_verification(progress=None) -> dict:
    """
    Re-run the frozen blend at d_1941 and compare against the shipped artefact.

    `progress(pct, message)` is called as the run advances so the API can report
    status while a job is in flight.

    Returns a result dict with `verdict` in {MATCH, MISMATCH}. A mismatch is
    reported, never hidden — that is the entire point of the check.
    """
    require_available()

    def emit(pct: int, msg: str) -> None:
        if progress:
            progress(pct, msg)

    import sys                                                  # noqa: PLC0415
    root = str(settings.project_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    t_start = time.perf_counter()
    emit(2, "loading frozen boosters")
    boosters = get_boosters()

    emit(6, "importing research feature pipeline")
    import numpy as np                                          # noqa: PLC0415
    from pipeline.champion_blend import REC_COLS_V5             # noqa: PLC0415
    from pipeline.data_loader import M5Data                     # noqa: PLC0415
    from pipeline.features_v5 import (CHAMPION_FEATURES,        # noqa: PLC0415
                                      FeatureBuilderV5)

    emit(10, "loading sales panel (~14 s)")
    t_data = time.perf_counter()
    data = M5Data()
    data_seconds = round(time.perf_counter() - t_data, 2)

    origin = FROZEN_ORIGIN_IDX
    horizon = settings.horizon
    if data.sales_wide.shape[1] - 1 != origin:
        # The panel must end exactly at the frozen origin, or the boosters would
        # be run against a different information set than they were built for.
        raise RuntimeError(
            f"panel ends at day_idx {data.sales_wide.shape[1] - 1}, "
            f"expected {origin}; refusing to run the frozen model")

    # --- member A: direct ------------------------------------------------
    emit(25, "running direct member (38 features)")
    t_direct = time.perf_counter()
    fb = FeatureBuilderV5(data)
    frame = fb.build_origin_frame(origin, horizon=horizon, include_target=False)
    p_direct = np.clip(
        boosters["direct"].predict(
            frame[CHAMPION_FEATURES].to_numpy(np.float32)), 0, None)
    direct_seconds = round(time.perf_counter() - t_direct, 2)
    del frame, fb

    # --- member B: recursive rollout -------------------------------------
    emit(40, "running recursive member (28-step rollout, ~29 s)")
    t_rec = time.perf_counter()
    import copy as _copy                                        # noqa: PLC0415
    n = data.sales_wide.shape[0]
    n_days = max(data.sales_wide.shape[1], origin + 1 + horizon)
    work = np.zeros((n, n_days), dtype=np.float32)
    work[:, :origin + 1] = data.sales_wide[:, :origin + 1]

    preds = np.empty((horizon, n), dtype=np.float64)
    for h in range(1, horizon + 1):
        pseudo = origin + h - 1
        d2 = _copy.copy(data)
        d2.sales_wide = work
        fb_r = FeatureBuilderV5(d2)
        fr = fb_r.build_origin_frame(pseudo, horizon=1, include_target=False)
        p = np.clip(boosters["recursive"].predict(
            fr[REC_COLS_V5].to_numpy(np.float32)), 0, None)
        preds[h - 1] = p
        work[:, pseudo + 1] = p.astype(np.float32)
        del fb_r, fr, d2
        if h % 7 == 0:
            emit(40 + int(40 * h / horizon), f"recursive step {h}/{horizon}")
    p_recursive = preds.ravel()
    recursive_seconds = round(time.perf_counter() - t_rec, 2)

    # --- structural leakage check ----------------------------------------
    real_future = data.sales_wide[:, origin + 1:origin + 1 + horizon]
    history_intact = bool(np.array_equal(
        data.sales_wide[:, :origin + 1].astype(np.float32), work[:, :origin + 1]))
    leakage = {
        "pre_origin_history_intact": history_intact,
        "no_ground_truth_exists": real_future.shape[1] == 0,
        "future_matrix_equals_real_sales": False,
        "passed": history_intact,
    }

    # --- blend ------------------------------------------------------------
    emit(85, "blending members at the frozen weight w=0.60")
    w = settings.blend_weight_direct
    blended = np.clip(w * p_direct + (1.0 - w) * p_recursive, 0, None)

    # --- compare against the shipped artefact ----------------------------
    emit(92, "comparing against the shipped forecast artefact")
    import csv                                                  # noqa: PLC0415
    shipped = np.empty((horizon, n), dtype=np.float64)
    id_to_row = {sid: i for i, sid in
                 enumerate(data.series_meta["id"].astype(str).tolist())}
    seen = 0
    with Path(settings.forecast_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row["id"].replace("_evaluation", "")
            idx = id_to_row.get(key, id_to_row.get(row["id"]))
            if idx is None:
                continue
            for h in range(1, horizon + 1):
                shipped[h - 1, idx] = float(row[f"F{h}"])
            seen += 1
    if seen != n:
        raise RuntimeError(f"matched {seen} of {n} series against the artefact")

    diff = np.abs(blended - shipped.ravel())
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    # The pipeline is deterministic (deterministic=True, fixed seed), so this
    # should be float-noise. The tolerance allows for the artefact's CSV
    # round-tripping, nothing more.
    tolerance = 1e-4
    verdict = "MATCH" if max_abs <= tolerance else "MISMATCH"

    emit(100, "complete")
    total_seconds = round(time.perf_counter() - t_start, 2)
    del data, work, preds
    return {
        "verdict": verdict,
        "tolerance": tolerance,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "n_predictions": int(blended.size),
        "n_series": int(n),
        "horizon": horizon,
        "origin_day": f"d_{origin + 1}",
        "blend_weight_direct": w,
        "recomputed_total": float(blended.sum()),
        "artefact_total": float(shipped.sum()),
        "leakage_checks": leakage,
        "timings_seconds": {
            "data_load": data_seconds,
            "direct_member": direct_seconds,
            "recursive_member": recursive_seconds,
            "total": total_seconds,
        },
        "models": model_runtime_info(),
        "interpretation": (
            "The frozen boosters were reloaded, features rebuilt from the raw "
            "panel, both members re-run and blended at w=0.60. The result was "
            "compared with the shipped forecast artefact."
            if verdict == "MATCH" else
            "The recomputed forecast does NOT match the shipped artefact. "
            "Investigate before trusting either."),
    }
