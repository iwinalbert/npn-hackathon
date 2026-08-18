
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
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment
from pipeline.features_v3 import FeatureBuilderV3, V3_FEATURES, feature_set

CHAMP_RMSE, CHAMP_MAE = 2.1210429411947650, 1.0319268155496617
PROMOTE = -0.010


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    t0 = time.time()
    banner("EXPERIMENT #71 — YEAR-OVER-YEAR FEATURES")
    cols = feature_set()
    print(f"  champion 32 features + {len(V3_FEATURES)} new = {len(cols)}")
    print(f"  new: {V3_FEATURES}\n")

    s = optimize.Setup()
    s.fb = FeatureBuilderV3(s.data)
    s.bt = Backtester(s.data, feature_builder=s.fb)
    s.valid = s.bt.build_validation_frame(s.origin_idx)
    s.y = s.valid["sales"].to_numpy()
    y = s.y

    banner("LEAKAGE TEST")
    clean = s.fb.build_origin_frame(s.origin_idx, include_target=False)
    corrupt = s.data.sales_wide.copy()
    corrupt[:, s.origin_idx + 1:] = 9999
    d2 = copy.copy(s.data); d2.sales_wide = corrupt
    dirty = FeatureBuilderV3(d2).build_origin_frame(s.origin_idx, include_target=False)
    changed = [c for c in cols
               if not np.array_equal(clean[c].to_numpy(), dirty[c].to_numpy(),
                                     equal_nan=np.issubdtype(clean[c].dtype, np.floating))]
    print(f"  {'PASS' if not changed else 'FAIL'} — {len(cols)} features, "
          f"{len(changed)} changed when every post-origin sale is set to 9999")
    if changed:
        raise SystemExit(f"STOP: leakage in {changed}")
    del clean, dirty, corrupt, d2

    f = s.valid
    print("\n  new feature sanity (validation frame):")
    for c in V3_FEATURES:
        v = f[c].to_numpy()
        fin = v[np.isfinite(v)]
        print(f"    {c:<26} missing {np.isnan(v).mean()*100:5.2f}%  "
              f"range [{fin.min():.3f}, {fin.max():.3f}]  mean {fin.mean():.3f}")

    banner("TRAIN (identical config to champion, only the feature set differs)")
    X, Y = optimize.build_matrix(s, cols, verbose=True)
    booster, info = optimize.train(X, Y, cols)
    del X, Y
    p = optimize.predict(booster, s, cols)
    d = optimize.diagnostics(y, p, s)
    dr, dm = d["RMSE"] - CHAMP_RMSE, d["MAE"] - CHAMP_MAE

    print(f"\n  champion : RMSE {CHAMP_RMSE:.4f}  MAE {CHAMP_MAE:.4f}")
    print(f"  with YoY : RMSE {d['RMSE']:.4f}  MAE {d['MAE']:.4f}")
    print(f"  change   : dRMSE {dr:+.4f}   dMAE {dm:+.4f}")
    print(f"  high-volume RMSE {d['high_volume_RMSE']:.4f} (champion 5.9756)")

    banner("DID THE MODEL USE THE NEW FEATURES?")
    gain = booster.feature_importance("gain")
    names = booster.feature_name()
    imp = pd.DataFrame({"feature": names, "gain": gain})
    imp["pct"] = imp.gain / imp.gain.sum() * 100
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    newshare = float(imp[imp.feature.isin(V3_FEATURES)].pct.sum())
    print(f"  combined gain share of the 4 new features: {newshare:.2f}%")
    for _, r in imp[imp.feature.isin(V3_FEATURES)].iterrows():
        rank = int(imp[imp.feature == r.feature].index[0]) + 1
        print(f"    {r['feature']:<26} {r['pct']:5.2f}%  (rank {rank} of {len(imp)})")
    print("\n  top 5 overall:")
    for _, r in imp.head(5).iterrows():
        print(f"    {r['feature']:<26} {r['pct']:5.2f}%")

    decision = "PROMOTE" if dr <= PROMOTE else "REJECT"
    banner("DECISION")
    print(f"  criterion: promote if dRMSE <= {PROMOTE}")
    print(f"  measured : dRMSE {dr:+.4f}")
    print(f"  -> {decision}")

    exp = Experiment("exp_71_year_over_year_features", model_type="LightGBM",
                     objective="tweedie (variance_power=1.1)",
                     feature_set_label=f"Champion 32 + {len(V3_FEATURES)} YoY features",
                     n_features=len(cols), features=cols, **s.describe())
    exp.note("Experiment #71. Prompted by Exp #70's finding that six diverse "
             "models have residuals correlated at 0.9897 — model variance is not "
             "the lever, so only new information can help.")
    exp.note("Materially different from the Phase 2 feature tests: those re-encoded "
             "information already present, whereas a 364-day lookback is outside "
             "the champion's entire feature horizon (max 28 days).")
    exp.note("Leakage corruption test passed on the new builder: all features "
             "unchanged when every post-origin sale is overwritten with 9999.")
    exp.set_metrics(**d)
    exp.set(hyperparameters=info["params"], n_estimators=info["n_estimators"],
            training_seconds=info["training_seconds"],
            delta_rmse_vs_best=round(dr, 6), delta_mae_vs_best=round(dm, 6),
            new_feature_gain_share_pct=round(newshare, 3),
            new_feature_importance=imp[imp.feature.isin(V3_FEATURES)].to_dict("records"),
            top_features=imp.head(8).to_dict("records"),
            decision=decision)
    exp.save()

    pd.DataFrame({"series_idx": s.valid["series_idx"].to_numpy(),
                  "target_day_idx": s.valid["target_day_idx"].to_numpy(),
                  "horizon": s.valid["horizon"].to_numpy(),
                  "y_true": y, "y_pred": np.round(p, 5)}).to_csv(
        config.PREDICTIONS_DIR / "exp_71_yoy_validation.csv", index=False)
    (config.ARTIFACTS_DIR / "exp71_summary.json").write_text(json.dumps({
        "champion": {"RMSE": CHAMP_RMSE, "MAE": CHAMP_MAE},
        "with_yoy": d, "delta": {"RMSE": dr, "MAE": dm},
        "new_feature_gain_share_pct": newshare,
        "new_feature_importance": imp[imp.feature.isin(V3_FEATURES)].to_dict("records"),
        "decision": decision}, indent=2, default=str), encoding="utf-8")
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
