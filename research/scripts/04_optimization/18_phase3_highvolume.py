
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics, optimize
from pipeline.features_v2 import V2_SETS


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def residual_table(df: pd.DataFrame, col: str, y, p) -> pd.DataFrame:
    g = df.groupby(col, observed=True)
    out = g.apply(lambda d: pd.Series({
        "n": len(d),
        "actual_mean": d["y"].mean(),
        "pred_mean": d["p"].mean(),
        "bias": d["p"].mean() - d["y"].mean(),
        "RMSE": float(np.sqrt(((d["y"] - d["p"]) ** 2).mean())),
        "MAE": float((d["y"] - d["p"]).abs().mean()),
        "sq_err_share_pct": ((d["y"] - d["p"]) ** 2).sum() / ((y - p) ** 2).sum() * 100,
    }), include_groups=False).reset_index()
    return out.sort_values("sq_err_share_pct", ascending=False)


def main():
    t0 = time.time()
    s = optimize.Setup()
    cols = V2_SETS["v2_base"]

    banner("PHASE 3A — DIAGNOSIS (no training)")
    P = pd.read_csv(config.PREDICTIONS_DIR /
                    "model_04_tweedie_recency_listing_validation.csv")
    P = P.sort_values(["target_day_idx", "series_idx"]).reset_index(drop=True)
    V = s.valid.sort_values(["target_day_idx", "series_idx"]).reset_index(drop=True)
    assert np.array_equal(P["y_true"].to_numpy(), V["sales"].to_numpy())

    y = P["y_true"].to_numpy(float)
    p = P["y_pred"].to_numpy(float)
    meta = s.data.series_meta
    si = P["series_idx"].to_numpy()

    hist = s.data.sales_wide[:, :s.origin_idx + 1]
    hmean = hist.mean(axis=1)[si]

    df = pd.DataFrame({
        "y": y, "p": p,
        "cat": meta["cat_id"].to_numpy()[si],
        "store": meta["store_id"].to_numpy()[si],
        "dept": meta["dept_id"].to_numpy()[si],
        "wday": s.data.calendar["weekday"].to_numpy()[P["target_day_idx"]],
        "is_event": (~s.data.calendar["event_name_1"].isna()).to_numpy()[P["target_day_idx"]],
        "volume_tier": pd.cut(hmean, [-0.001, 0.2, 1.0, 3.0, np.inf],
                              labels=["very low", "low", "medium", "high"]),
    })
    price = V["sell_price"].to_numpy()
    df["price_regime"] = pd.cut(price, [-0.001, 1, 3, 6, np.inf],
                                labels=["<$1", "$1-3", "$3-6", ">$6"])

    diagnostics = {}
    for col in ["volume_tier", "cat", "dept", "store", "wday", "is_event", "price_regime"]:
        t = residual_table(df, col, y, p)
        diagnostics[col] = t.to_dict(orient="records")
        print(f"\n--- residuals by {col} ---")
        print(t.head(6).to_string(index=False,
              float_format=lambda v: f"{v:8.3f}"))

    item_sq = pd.DataFrame({"item": meta["item_id"].to_numpy()[si],
                            "sq": (y - p) ** 2})
    top_items = item_sq.groupby("item")["sq"].sum().sort_values(ascending=False)
    share_top50 = top_items.head(50).sum() / item_sq["sq"].sum() * 100
    print(f"\n  top 50 items (of 3,049) carry {share_top50:.1f}% of all squared error")
    diagnostics["top50_items_sq_error_share_pct"] = round(float(share_top50), 2)

    print("\n--- actual vs predicted, by volume tier ---")
    vt = residual_table(df, "volume_tier", y, p)
    print(vt.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    (config.ARTIFACTS_DIR / "phase3_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")

    banner("PHASE 3B — LEGITIMATE FIXES")
    results = []

    print("  Building volume weights (capped at 5x)...")
    hist_mean_all = s.data.sales_wide[:, :s.origin_idx + 1].mean(axis=1)

    def weights_for(origins, cap):
        w = []
        for _ in origins:
            v = np.clip(1.0 + hist_mean_all, 1.0, cap)
            w.append(np.tile(v, config.HORIZON))
        return np.concatenate(w).astype(np.float32)

    for cap, tag in [(3.0, "cap3"), (5.0, "cap5")]:
        X, Y = optimize.build_matrix(s, cols)
        w = weights_for(s.origins, cap)
        assert len(w) == len(Y)
        booster, info = optimize.train(X, Y, cols, weights=w)
        del X, Y, w
        pr = optimize.predict(booster, s, cols)
        d = optimize.diagnostics(s.y, pr, s)
        from pipeline.experiment import Experiment
        e = Experiment(f"opt_03_volume_weight_{tag}", model_type="LightGBM",
                       objective="tweedie (variance_power=1.1)",
                       feature_set_label=f"Volume-weighted training (cap {cap}x)",
                       n_features=len(cols), **s.describe())
        e.note("Phase 3. Training rows weighted by the series' own historical "
               "mean demand, capped, so high-volume series carry more of the "
               "loss. Features and objective unchanged.")
        e.set_metrics(**d)
        e.set(hyperparameters=info["params"], n_estimators=info["n_estimators"],
              training_seconds=info["training_seconds"], weight_cap=cap,
              delta_rmse_vs_best=round(d["RMSE"] - optimize.BEST_RMSE, 6),
              delta_mae_vs_best=round(d["MAE"] - optimize.BEST_MAE, 6))
        e.save()
        print(f"  [volume weight cap {cap}] RMSE={d['RMSE']:.4f} "
              f"({d['RMSE']-optimize.BEST_RMSE:+.4f})  MAE={d['MAE']:.4f} "
              f"({d['MAE']-optimize.BEST_MAE:+.4f})  hv={d['high_volume_RMSE']:.3f}")
        results.append({"experiment": f"volume weight cap {cap}x", **d})
        del booster

    print("\n  Post-hoc high-volume calibration (factor chosen on inner window)...")
    inner = optimize.Setup(origin_idx=config.VALIDATION_ORIGIN_IDX - config.HORIZON)
    Xi, Yi = optimize.build_matrix(inner, cols)
    bi, _ = optimize.train(Xi, Yi, cols)
    del Xi, Yi
    pi = optimize.predict(bi, inner, cols)
    best_f, best_r = 1.0, metrics.rmse(inner.y, pi)
    for f in np.arange(1.00, 1.31, 0.02):
        q = pi.copy()
        q[inner.high] *= f
        r = metrics.rmse(inner.y, q)
        if r < best_r:
            best_f, best_r = float(f), r
    print(f"    inner-window best factor: {best_f:.2f} (inner RMSE {best_r:.4f} "
          f"vs {metrics.rmse(inner.y, pi):.4f} unscaled)")
    del bi, pi, inner

    q = p.copy()
    q[s.high] *= best_f
    dcal = optimize.diagnostics(s.y, q, s)
    from pipeline.experiment import Experiment
    e = Experiment("opt_03_highvol_calibration", model_type="post-hoc calibration",
                   objective="n/a (rescaling of model_04 predictions)",
                   feature_set_label=f"High-volume predictions x{best_f:.2f}",
                   n_features=len(cols), **s.describe())
    e.note(f"Phase 3. Multiplier {best_f:.2f} chosen on the inner window only, "
           "then applied once here. No retraining.")
    e.set_metrics(**dcal)
    e.set(calibration_factor=best_f, training_seconds=0,
          delta_rmse_vs_best=round(dcal["RMSE"] - optimize.BEST_RMSE, 6),
          delta_mae_vs_best=round(dcal["MAE"] - optimize.BEST_MAE, 6))
    e.save()
    print(f"  [calibration x{best_f:.2f}] RMSE={dcal['RMSE']:.4f} "
          f"({dcal['RMSE']-optimize.BEST_RMSE:+.4f})  MAE={dcal['MAE']:.4f} "
          f"({dcal['MAE']-optimize.BEST_MAE:+.4f})")
    results.append({"experiment": f"high-vol calibration x{best_f:.2f}", **dcal})

    banner("PHASE 3 SUMMARY")
    out = pd.DataFrame([{
        "experiment": r["experiment"], "RMSE": r["RMSE"], "MAE": r["MAE"],
        "dRMSE": r["RMSE"] - optimize.BEST_RMSE,
        "dMAE": r["MAE"] - optimize.BEST_MAE,
        "high_vol_RMSE": r["high_volume_RMSE"],
        "high_vol_bias": r["high_volume_bias"],
    } for r in results])
    base = pd.DataFrame([{"experiment": "current best (reference)",
                          "RMSE": optimize.BEST_RMSE, "MAE": optimize.BEST_MAE,
                          "dRMSE": 0.0, "dMAE": 0.0,
                          "high_vol_RMSE": 5.9756, "high_vol_bias": -0.389}])
    out = pd.concat([base, out], ignore_index=True)
    print(out.to_string(index=False))
    out.to_csv(config.ARTIFACTS_DIR / "phase3_highvolume_results.csv", index=False)
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
