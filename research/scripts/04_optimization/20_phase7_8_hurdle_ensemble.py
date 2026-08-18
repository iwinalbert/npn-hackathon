
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
from pipeline.experiment import Experiment
from pipeline.features_v2 import V2_SETS

COLS = V2_SETS["v2_base"]
INNER = config.VALIDATION_ORIGIN_IDX - config.HORIZON


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def hurdle_fit_predict(s, stage2_params, label):
    X, Y = optimize.build_matrix(s, COLS)
    ybin = (Y > 0).astype(np.float32)
    pos = Y > 0
    b1, i1 = optimize.train(X, ybin, COLS, params={"objective": "binary",
                                                   "metric": "binary_logloss"})
    b2, i2 = optimize.train(X[pos], Y[pos], COLS, params=stage2_params)
    del X, Y, ybin, pos
    Xv = s.valid[COLS].to_numpy(np.float32)
    prob = np.clip(b1.predict(Xv), 0, 1)
    mag = np.clip(b2.predict(Xv), 0, None)
    secs = i1["training_seconds"] + i2["training_seconds"]
    return prob, mag, secs, (b1, b2)


def main():
    t0 = time.time()
    s = optimize.Setup()
    results = []

    banner("PHASE 7 — HURDLE, SECOND ATTEMPT")
    print("  Original hurdle used Poisson for stage 2 and scored 2.1267.")
    print("  Variant here: Tweedie for stage 2, plus a calibration factor on the")
    print("  product P(sale) x E[units|sale], chosen on the INNER window.\n")

    prob, mag, secs, _ = hurdle_fit_predict(
        s, {"objective": "tweedie", "tweedie_variance_power": 1.1},
        "stage2 tweedie")
    raw = prob * mag
    d_raw = optimize.diagnostics(s.y, raw, s)
    print(f"  [hurdle v2, uncalibrated] RMSE={d_raw['RMSE']:.4f} "
          f"({d_raw['RMSE']-optimize.BEST_RMSE:+.4f})  MAE={d_raw['MAE']:.4f}")
    print(f"    mean P(sale)={prob.mean():.4f}  mean E[units|sale]={mag.mean():.4f}")

    print("\n  Choosing a calibration factor on the inner window...")
    si = optimize.Setup(origin_idx=INNER)
    ip, im, _, _ = hurdle_fit_predict(
        si, {"objective": "tweedie", "tweedie_variance_power": 1.1}, "inner")
    iraw = ip * im
    best_f, best_r = 1.0, metrics.rmse(si.y, iraw)
    for f in np.arange(0.80, 1.41, 0.02):
        r = metrics.rmse(si.y, iraw * f)
        if r < best_r:
            best_f, best_r = float(f), r
    print(f"    factor {best_f:.2f} (inner RMSE {best_r:.4f} vs "
          f"{metrics.rmse(si.y, iraw):.4f} uncalibrated)")
    del si, ip, im, iraw

    cal = raw * best_f
    d_cal = optimize.diagnostics(s.y, cal, s)
    print(f"  [hurdle v2, calibrated x{best_f:.2f}] RMSE={d_cal['RMSE']:.4f} "
          f"({d_cal['RMSE']-optimize.BEST_RMSE:+.4f})  MAE={d_cal['MAE']:.4f}")

    for tag, dd, extra in [("opt_07_hurdle_v2", d_raw, {}),
                           ("opt_07_hurdle_v2_calibrated", d_cal,
                            {"calibration_factor": best_f})]:
        e = Experiment(tag, model_type="LightGBM two-stage hurdle",
                       objective="stage1=binary, stage2=tweedie(1.1)",
                       feature_set_label="Hurdle v2 (32 features)",
                       n_features=len(COLS), **s.describe(), **extra)
        e.note("Phase 7. Second attempt at the hurdle: Tweedie replaces Poisson "
               "for the magnitude stage. Any calibration factor was chosen on the "
               "inner window only.")
        e.set_metrics(**dd)
        e.set(training_seconds=round(secs, 1),
              mean_predicted_probability=round(float(prob.mean()), 5),
              mean_predicted_magnitude=round(float(mag.mean()), 5),
              delta_rmse_vs_best=round(dd["RMSE"] - optimize.BEST_RMSE, 6),
              delta_mae_vs_best=round(dd["MAE"] - optimize.BEST_MAE, 6))
        e.save()
        results.append({"model": tag, **dd})

    pd.DataFrame({
        "series_idx": s.valid["series_idx"].to_numpy(),
        "target_day_idx": s.valid["target_day_idx"].to_numpy(),
        "horizon": s.valid["horizon"].to_numpy(),
        "y_true": s.y, "y_pred": np.round(cal, 5),
    }).to_csv(config.PREDICTIONS_DIR / "opt_07_hurdle_v2_validation.csv", index=False)

    best_hurdle = min(d_raw["RMSE"], d_cal["RMSE"])
    print(f"\n  VERDICT: best hurdle RMSE {best_hurdle:.4f} vs single-model "
          f"{optimize.BEST_RMSE:.4f} -> "
          f"{'hurdle wins' if best_hurdle < optimize.BEST_RMSE else 'hurdle still loses'}")
    del prob, mag, raw, cal

    banner("PHASE 8 — ENSEMBLE (weights chosen on the inner window)")
    print("  Candidates are the two objectives that won on different metrics:")
    print("  Tweedie (best RMSE) and L1 (best MAE). Blending them tests whether")
    print("  the RMSE/MAE trade can be improved rather than just moved.\n")

    print("  Training both on the inner window to pick a weight...")
    si = optimize.Setup(origin_idx=INNER)
    Xi, Yi = optimize.build_matrix(si, COLS)
    bt_i, _ = optimize.train(Xi, Yi, COLS)
    pl_i = None
    bl_i, _ = optimize.train(Xi, Yi, COLS, params={"objective": "regression_l1"})
    del Xi, Yi
    pt_i = optimize.predict(bt_i, si, COLS)
    pl_i = optimize.predict(bl_i, si, COLS)
    del bt_i, bl_i

    grid = np.arange(0.0, 1.01, 0.1)
    rows = []
    for w in grid:
        b = w * pt_i + (1 - w) * pl_i
        rows.append({"w_tweedie": round(float(w), 2),
                     "inner_RMSE": metrics.rmse(si.y, b),
                     "inner_MAE": metrics.mae(si.y, b)})
    gdf = pd.DataFrame(rows)
    print(gdf.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    w_best = float(gdf.sort_values("inner_RMSE").iloc[0]["w_tweedie"])
    print(f"\n  weight chosen on inner window: {w_best:.2f} Tweedie / "
          f"{1-w_best:.2f} L1")
    del si, pt_i, pl_i

    pt = pd.read_csv(config.PREDICTIONS_DIR / "opt_00_baseline_reproduce_validation.csv")
    pl = pd.read_csv(config.PREDICTIONS_DIR / "opt_06_obj_l1_validation.csv")
    for df in (pt, pl):
        df.sort_values(["target_day_idx", "series_idx"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    assert np.array_equal(pt["y_true"].to_numpy(), pl["y_true"].to_numpy())
    y = pt["y_true"].to_numpy(float)
    blend = w_best * pt["y_pred"].to_numpy(float) + (1 - w_best) * pl["y_pred"].to_numpy(float)

    order = s.valid.sort_values(["target_day_idx", "series_idx"]).index
    s_sorted = s.valid.loc[order]
    high_sorted = s.high[order.to_numpy()]

    d_ens = metrics.evaluate(y, blend)
    hi = metrics.evaluate(y[high_sorted], blend[high_sorted])
    print(f"\n  ENSEMBLE on primary: RMSE={d_ens['RMSE']:.4f} "
          f"({d_ens['RMSE']-optimize.BEST_RMSE:+.4f})  MAE={d_ens['MAE']:.4f} "
          f"({d_ens['MAE']-optimize.BEST_MAE:+.4f})  highvol RMSE={hi['RMSE']:.3f}")

    e = Experiment("opt_08_ensemble_tweedie_l1", model_type="weighted ensemble",
                   objective=f"{w_best:.2f}*tweedie + {1-w_best:.2f}*L1",
                   feature_set_label="Ensemble of two objectives, 32 features",
                   n_features=len(COLS), **s.describe())
    e.note("Phase 8. Blend weight selected on the inner window d_1886..d_1913 and "
           "applied once here, so the primary window remains unbiased.")
    e.set_metrics(**d_ens, high_volume_RMSE=hi["RMSE"], high_volume_bias=hi["bias"])
    e.set(ensemble_weight_tweedie=w_best, training_seconds=0,
          weight_grid=gdf.to_dict(orient="records"),
          delta_rmse_vs_best=round(d_ens["RMSE"] - optimize.BEST_RMSE, 6),
          delta_mae_vs_best=round(d_ens["MAE"] - optimize.BEST_MAE, 6))
    e.save()

    pd.DataFrame({"series_idx": pt["series_idx"], "target_day_idx": pt["target_day_idx"],
                  "y_true": y, "y_pred": np.round(blend, 5)}).to_csv(
        config.PREDICTIONS_DIR / "opt_08_ensemble_validation.csv", index=False)

    (config.ARTIFACTS_DIR / "phase7_8_summary.json").write_text(json.dumps({
        "hurdle_raw": d_raw, "hurdle_calibrated": d_cal,
        "hurdle_calibration_factor": best_f,
        "ensemble": {**d_ens, "high_volume_RMSE": hi["RMSE"],
                     "weight_tweedie": w_best},
        "ensemble_weight_grid": gdf.to_dict(orient="records"),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
