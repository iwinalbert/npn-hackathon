
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics, optimize
from pipeline.backtest import Backtester
from pipeline.experiment import Experiment
from pipeline.features_v4 import FeatureBuilderV4, V4_FEATURES, feature_set

CHAMP_RMSE, CHAMP_MAE = 2.1210429411947650, 1.0319268155496617
PROMOTE = -0.010


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def build_setup(origin_idx):
    s = optimize.Setup(origin_idx=origin_idx)
    s.fb = FeatureBuilderV4(s.data)
    s.bt = Backtester(s.data, feature_builder=s.fb)
    s.valid = s.bt.build_validation_frame(origin_idx)
    s.y = s.valid["sales"].to_numpy()
    return s


def main():
    t0 = time.time()
    banner("EXPERIMENT #72 — PER-SERIES RESPONSE-SHAPE FEATURES")
    cols = feature_set()
    print(f"  champion 32 features + {len(V4_FEATURES)} shape features = {len(cols)}")
    print(f"  new: {V4_FEATURES}\n")

    s = build_setup(config.VALIDATION_ORIGIN_IDX)

    banner("LEAKAGE TEST")
    clean = s.fb.build_origin_frame(s.origin_idx, include_target=False)
    corrupt = s.data.sales_wide.copy()
    corrupt[:, s.origin_idx + 1:] = 9999
    d2 = copy.copy(s.data); d2.sales_wide = corrupt
    dirty = FeatureBuilderV4(d2).build_origin_frame(s.origin_idx, include_target=False)
    changed = [c for c in cols
               if not np.array_equal(clean[c].to_numpy(), dirty[c].to_numpy(),
                                     equal_nan=np.issubdtype(clean[c].dtype, np.floating))]
    print(f"  {'PASS' if not changed else 'FAIL'} — {len(cols)} features, "
          f"{len(changed)} changed when post-origin sales set to 9999")
    if changed:
        raise SystemExit(f"STOP: leakage in {changed}")
    del clean, dirty, corrupt, d2

    print("\n  shape-feature sanity:")
    for c in V4_FEATURES:
        v = s.valid[c].to_numpy()
        print(f"    {c:<16} range [{v.min():.3f}, {v.max():.3f}]  "
              f"mean {v.mean():.4f}  sd {v.std():.4f}")

    banner("TRAIN (identical config to champion; only the feature set differs)")
    X, Y = optimize.build_matrix(s, cols, verbose=True)
    booster, info = optimize.train(X, Y, cols)
    del X, Y
    p = optimize.predict(booster, s, cols)
    d = optimize.diagnostics(s.y, p, s)
    dr, dm = d["RMSE"] - CHAMP_RMSE, d["MAE"] - CHAMP_MAE

    print(f"\n  champion   : RMSE {CHAMP_RMSE:.4f}  MAE {CHAMP_MAE:.4f}")
    print(f"  with shape : RMSE {d['RMSE']:.4f}  MAE {d['MAE']:.4f}")
    print(f"  change     : dRMSE {dr:+.4f}   dMAE {dm:+.4f}")
    print(f"  high-volume RMSE {d['high_volume_RMSE']:.4f} (champion 5.9756)")

    gain = booster.feature_importance("gain")
    imp = pd.DataFrame({"feature": booster.feature_name(), "gain": gain})
    imp["pct"] = imp.gain / imp.gain.sum() * 100
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    newshare = float(imp[imp.feature.isin(V4_FEATURES)].pct.sum())
    print(f"\n  shape features took {newshare:.2f}% of total gain:")
    for _, r in imp[imp.feature.isin(V4_FEATURES)].iterrows():
        rank = int(imp[imp.feature == r.feature].index[0]) + 1
        print(f"    {r['feature']:<16} {r['pct']:5.2f}%  (rank {rank}/{len(imp)})")

    decision = "PROMOTE" if dr <= PROMOTE else "REJECT"
    print(f"\n  criterion: promote if dRMSE <= {PROMOTE}   ->  {decision}")

    exp = Experiment("exp_72_per_series_shape_features", model_type="LightGBM",
                     objective="tweedie (variance_power=1.1)",
                     feature_set_label=f"Champion 32 + {len(V4_FEATURES)} shape features",
                     n_features=len(cols), features=cols, **s.describe())
    exp.note("Experiment #72. Motivated by diagnostic 33: an arithmetic "
             "level x weekday-ratio predictor beats level-only by 0.0578 out of "
             "sample, so per-series weekly shape is real signal.")
    exp.note("Materially different from Phase 2 and Experiment #71, which tested "
             "LEVEL features only. These describe shape relative to a series' own "
             "average.")
    exp.note("Leakage corruption test passed on the new builder.")
    exp.set_metrics(**d)
    exp.set(hyperparameters=info["params"], n_estimators=info["n_estimators"],
            training_seconds=info["training_seconds"],
            delta_rmse_vs_best=round(dr, 6), delta_mae_vs_best=round(dm, 6),
            new_feature_gain_share_pct=round(newshare, 3),
            new_feature_importance=imp[imp.feature.isin(V4_FEATURES)].to_dict("records"),
            decision=decision)
    exp.save()

    pd.DataFrame({"series_idx": s.valid["series_idx"].to_numpy(),
                  "target_day_idx": s.valid["target_day_idx"].to_numpy(),
                  "horizon": s.valid["horizon"].to_numpy(),
                  "y_true": s.y, "y_pred": np.round(p, 5)}).to_csv(
        config.PREDICTIONS_DIR / "exp_72_shape_validation.csv", index=False)

    summary = {"champion": {"RMSE": CHAMP_RMSE, "MAE": CHAMP_MAE},
               "with_shape": d, "delta": {"RMSE": dr, "MAE": dm},
               "gain_share_pct": newshare, "decision": decision}

    if decision == "PROMOTE":
        banner("ROBUSTNESS — three further windows (champion vs shape)")
        cal = s.data.calendar
        dates = pd.to_datetime(cal["date"])
        idx = lambda ds: int(cal.index[dates == pd.Timestamp(ds)][0])
        windows = {"christmas_2015": idx("2015-12-25") - 14,
                   "summer_2015": idx("2015-07-15"),
                   "autumn_2015": idx("2015-10-01")}
        rows = []
        from pipeline.features_v2 import V2_SETS
        base_cols = V2_SETS["v2_base"]
        for wname, o in windows.items():
            sw = build_setup(o)
            Xs, Ys = optimize.build_matrix(sw, cols)
            bs, _ = optimize.train(Xs, Ys, cols)
            ps = optimize.predict(bs, sw, cols); del Xs, Ys, bs
            Xb, Yb = optimize.build_matrix(sw, base_cols)
            bb, _ = optimize.train(Xb, Yb, base_cols)
            pb = optimize.predict(bb, sw, base_cols); del Xb, Yb, bb
            rs, rb = metrics.rmse(sw.y, ps), metrics.rmse(sw.y, pb)
            print(f"  {wname:<18} champion {rb:.4f}   shape {rs:.4f}   ({rs-rb:+.4f})")
            rows.append({"window": wname, "champion_RMSE": rb, "shape_RMSE": rs,
                         "delta": rs - rb})
            del sw
        summary["robustness"] = rows
        wins = sum(1 for r in rows if r["delta"] < 0)
        print(f"\n  shape features improved {wins}/3 additional windows")
        summary["robustness_wins"] = wins

    (config.ARTIFACTS_DIR / "exp72_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
