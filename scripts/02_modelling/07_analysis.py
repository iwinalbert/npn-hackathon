"""
Phase 10: feature importance and error analysis for the best measured model.

The "best" model is chosen mechanically — lowest RMSE among every completed
experiment evaluated on the primary validation window — not by preference.

    python scripts/07_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

import lightgbm as lgb

from pipeline import config, experiment, metrics
from pipeline.data_loader import M5Data

PRIMARY_DAYS = "d_1914 .. d_1941"

# Plain-English glosses used in the reports.
FEATURE_GLOSSARY = {
    "lag_1": "sales on the forecast origin day (the last day we know about)",
    "lag_7": "sales 7 days before the first forecast day",
    "lag_14": "sales 14 days before the first forecast day",
    "lag_28": "sales 28 days before the first forecast day",
    "rolling_mean_7": "average daily sales over the last 7 days before the forecast",
    "rolling_mean_28": "average daily sales over the last 28 days before the forecast",
    "rolling_std_7": "how much daily sales bounced around over the last 7 days",
    "rolling_std_28": "how much daily sales bounced around over the last 28 days",
    "days_since_last_sale": "number of days since this item last recorded any sale",
    "zero_streak_length": "length of the current run of zero-sales days (same number as days_since_last_sale at a fixed origin)",
    "days_since_first_sale": "how long this item has been selling at all, in days",
    "days_since_first_listing": "days since this product first had a price on record in this store",
    "pre_listing": "flag: this product had no price on record yet on the forecast day",
    "sell_price": "the price of this product in this store on the forecast day",
    "recent_avg_price": "this product's own average price over the weeks before the forecast",
    "price_rel_to_recent_avg": "current price divided by the product's own recent average price",
    "price_is_missing": "flag: no price on record for this product-week",
    "wday": "day of the week (1 = Saturday in this dataset)",
    "month": "calendar month",
    "year": "calendar year",
    "is_weekend": "flag: Saturday or Sunday",
    "event_name_1": "which named holiday/event falls on this day, if any",
    "event_type_1": "the broad type of that event (Sporting, Cultural, National, Religious)",
    "event_name_2": "a second simultaneous event, which is rare",
    "event_type_2": "the type of that second event",
    "snap": "whether SNAP food-assistance benefits were usable in this store's state that day",
    "item_id": "which product this is",
    "dept_id": "which department the product belongs to",
    "cat_id": "which category (FOODS / HOBBIES / HOUSEHOLD)",
    "store_id": "which of the 10 stores",
    "state_id": "which of the 3 states (CA / TX / WI)",
    "horizon": "how many days ahead this particular prediction is (1 to 28)",
}


def pick_best() -> dict:
    """
    Lowest primary-window RMSE among deployable experiments.

    Ablation rungs are excluded: they exist to measure the contribution of a
    feature group, not to produce a shippable model, and they deliberately save
    neither a model file nor predictions. A saved model and prediction file are
    required so importance and error analysis have something to read.
    """
    best, best_rmse = None, float("inf")
    for r in experiment.load_all():
        if r.get("status") != "completed":
            continue
        if r.get("tuning_window") == "INNER":       # inner-window runs are not comparable
            continue
        if r["experiment_name"].startswith("ablation_"):
            continue
        if r.get("validation_days") != PRIMARY_DAYS:
            continue
        if not r.get("prediction_path") or not r.get("model_path"):
            continue
        m = r.get("metrics", {})
        if "RMSE" in m and m["RMSE"] < best_rmse:
            best, best_rmse = r, m["RMSE"]
    if best is None:
        raise SystemExit("no completed, deployable experiment found on the primary window")
    return best


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("PHASE 10 — FEATURE IMPORTANCE AND ERROR ANALYSIS")
    print("=" * 78)

    best = pick_best()
    name = best["experiment_name"]
    print(f"  best model by measured RMSE: {name}")
    print(f"    RMSE={best['metrics']['RMSE']:.4f}  MAE={best['metrics']['MAE']:.4f}")
    print(f"    features: {best.get('n_features')} ({best.get('feature_set')})")

    out: dict = {
        "best_model": name,
        "best_metrics": best["metrics"],
        "feature_set": best.get("feature_set"),
        "validation_days": best.get("validation_days"),
        "validation_dates": best.get("validation_dates"),
    }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    model_path = best.get("model_path")
    if model_path and "+" not in str(model_path):
        booster = lgb.Booster(model_file=str(config.PROJECT_ROOT / model_path))
        names = booster.feature_name()
        gain = booster.feature_importance("gain")
        split = booster.feature_importance("split")

        imp = pd.DataFrame({
            "feature": names,
            "gain": gain,
            "gain_pct": gain / gain.sum() * 100,
            "splits": split,
        }).sort_values("gain", ascending=False).reset_index(drop=True)
        imp["meaning"] = imp["feature"].map(FEATURE_GLOSSARY).fillna("")

        imp.to_csv(config.ARTIFACTS_DIR / "feature_importance.csv", index=False)
        out["feature_importance"] = imp.to_dict(orient="records")

        print("\n  Top 15 features by gain "
              "(gain = total improvement this feature contributed to the model):")
        for _, r in imp.head(15).iterrows():
            print(f"    {r['gain_pct']:5.2f}%  {r['feature']:<26} {r['meaning'][:60]}")
        print("\n  NOTE: importance shows what the model USED, not what CAUSES sales.")
    else:
        print("\n  (best model is multi-part; skipping single-booster importance)")
        out["feature_importance"] = None

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------
    pred_path = config.PROJECT_ROOT / best["prediction_path"]
    P = pd.read_csv(pred_path)
    data = M5Data(load_prices=False)
    meta = data.series_meta

    P["cat_id"] = meta["cat_id"].to_numpy()[P["series_idx"]]
    P["dept_id"] = meta["dept_id"].to_numpy()[P["series_idx"]]
    P["store_id"] = meta["store_id"].to_numpy()[P["series_idx"]]
    P["state_id"] = meta["state_id"].to_numpy()[P["series_idx"]]

    cal = data.calendar
    P["weekday"] = cal["weekday"].to_numpy()[P["target_day_idx"]]
    P["is_weekend"] = cal["is_weekend"].to_numpy()[P["target_day_idx"]]
    P["is_event"] = (~cal["event_name_1"].isna()).to_numpy()[P["target_day_idx"]]
    snapm = data.snap_matrix
    P["snap"] = snapm[P["target_day_idx"].to_numpy(),
                      data.snap_col_of_series[P["series_idx"].to_numpy()]]

    # Series behaviour measured on TRAINING history only (up to the origin), so
    # the segmentation itself cannot use validation-period information.
    hist = data.sales_wide[:, :config.VALIDATION_ORIGIN_IDX + 1]
    series_mean = hist.mean(axis=1)
    series_zero = (hist == 0).mean(axis=1)
    P["hist_mean"] = series_mean[P["series_idx"]]
    P["hist_zero_pct"] = series_zero[P["series_idx"]]

    P["volume_tier"] = pd.cut(
        P["hist_mean"], [-0.001, 0.2, 1.0, 3.0, np.inf],
        labels=["very low (<0.2/day)", "low (0.2-1)", "medium (1-3)", "high (>3)"])
    P["sparsity_band"] = pd.cut(
        P["hist_zero_pct"], [-0.001, 0.5, 0.75, 0.95, 1.001],
        labels=["<50% zeros", "50-75% zeros", "75-95% zeros", ">95% zeros"])

    err = P["y_pred"] - P["y_true"]
    P["abs_err"] = err.abs()
    P["sq_err"] = err ** 2

    def group_stats(col: str) -> pd.DataFrame:
        g = P.groupby(col, observed=True).agg(
            n=("y_true", "size"),
            actual_mean=("y_true", "mean"),
            pred_mean=("y_pred", "mean"),
            RMSE=("sq_err", lambda s: float(np.sqrt(s.mean()))),
            MAE=("abs_err", "mean"),
        ).reset_index()
        g["bias"] = g["pred_mean"] - g["actual_mean"]
        g["share_of_total_sq_err"] = (
            P.groupby(col, observed=True)["sq_err"].sum().to_numpy() / P["sq_err"].sum() * 100
        )
        return g.sort_values("RMSE", ascending=False)

    breakdowns = {}
    for col, title in [
        ("cat_id", "Category"), ("dept_id", "Department"), ("store_id", "Store"),
        ("state_id", "State"), ("volume_tier", "Historical volume tier"),
        ("sparsity_band", "Historical sparsity"), ("weekday", "Day of week"),
        ("is_weekend", "Weekend vs weekday"), ("is_event", "Event day vs ordinary day"),
        ("snap", "SNAP day vs non-SNAP day"), ("horizon", "Days ahead (horizon)"),
    ]:
        g = group_stats(col)
        breakdowns[col] = {"title": title, "rows": g.to_dict(orient="records")}
        print(f"\n  --- {title} ---")
        show = g if len(g) <= 12 else pd.concat([g.head(4), g.tail(3)])
        for _, r in show.iterrows():
            print(f"    {str(r[col])[:26]:<28} n={int(r['n']):>7,}  "
                  f"RMSE={r['RMSE']:6.3f}  MAE={r['MAE']:5.3f}  "
                  f"bias={r['bias']:+6.3f}  errshare={r['share_of_total_sq_err']:5.2f}%")

    out["error_breakdowns"] = breakdowns

    # Where does total squared error actually live?
    top1 = P.nlargest(int(len(P) * 0.01), "sq_err")["sq_err"].sum() / P["sq_err"].sum() * 100
    out["error_concentration"] = {
        "pct_of_squared_error_from_worst_1pct_of_rows": round(float(top1), 2),
        "zero_actual_rows_pct": round(float((P["y_true"] == 0).mean() * 100), 2),
        "mean_pred_on_zero_actual_rows": round(float(P.loc[P["y_true"] == 0, "y_pred"].mean()), 4),
        "mean_pred_on_positive_actual_rows": round(float(P.loc[P["y_true"] > 0, "y_pred"].mean()), 4),
        "mean_actual_on_positive_rows": round(float(P.loc[P["y_true"] > 0, "y_true"].mean()), 4),
    }
    print(f"\n  --- Error concentration ---")
    print(f"    worst 1% of rows carry {top1:.2f}% of all squared error")
    print(f"    {out['error_concentration']['zero_actual_rows_pct']}% of validation rows "
          f"have actual sales = 0")
    print(f"    mean prediction on those zero rows: "
          f"{out['error_concentration']['mean_pred_on_zero_actual_rows']}")
    print(f"    mean prediction where actual > 0: "
          f"{out['error_concentration']['mean_pred_on_positive_actual_rows']} "
          f"(actual mean {out['error_concentration']['mean_actual_on_positive_rows']})")

    path = config.ARTIFACTS_DIR / "error_analysis.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {path.relative_to(config.PROJECT_ROOT)}")
    print(f"  wrote artifacts/feature_importance.csv")
    print(f"  total wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
