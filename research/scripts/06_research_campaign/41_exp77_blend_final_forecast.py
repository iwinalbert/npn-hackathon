"""
EXPERIMENT #78 — retrain the accepted blend at the real forecast origin and
produce the 28-day deliverable for d_1942..d_1969.

THE CONFIGURATION BEING PROMOTED
--------------------------------
    member A   direct champion, 38 features (Experiments #72-#75)
    member B'  one-step recursive, 32 features = the Phase-5 architecture plus
               the champion's six shape/cycle features (Experiment #77)
    blend      w = 0.60 on member A

Both members and the weight come from accepted experiments:
  #76 established the direct+recursive blend    (4/4 windows, 3/3 seeds)
  #77 upgraded member B and accepted it         (D1 4/4, D2 -0.0042, D3 3/4)

The weight w = 0.60 was selected on an INNER window (origin d_1885, targets
d_1886..d_1913) which lies strictly before this forecast origin, so no
information from the forecast horizon influenced it. Across the four evaluation
windows w = 0.60 gave mean dRMSE -0.0242 against member A alone, matching
w = 0.50's -0.0240 while costing +0.0186 MAE instead of +0.0244.

Runs only if BOTH #76 and #77 were accepted; the script checks their registry
records and refuses otherwise, so it cannot quietly promote a rejected
configuration.

Every earlier artefact is left untouched. The champion model, the shape+cycle
model, and both previous forecasts stay exactly where they are — everything
written here uses a new name.

WHAT IS DIFFERENT AT THE FORECAST ORIGIN
----------------------------------------
There is no ground truth for d_1942..d_1969, so no accuracy number applies to
this file. The accuracy estimate for this configuration is the one Experiment
#77 measured on four held-out windows, and that is what gets recorded.

The recursive member has to roll 28 days past the end of the sales matrix here,
which is why pipeline/recursive.py sizes its working matrix by what the rollout
needs rather than by how much history exists.

    python scripts/06_research_campaign/41_exp77_blend_final_forecast.py
"""

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

from pipeline import config, optimize, recursive
from pipeline.backtest import Backtester
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment
from pipeline.features_v4 import V4_FEATURES
from pipeline.features_v5 import FeatureBuilderV5, CHAMPION_FEATURES, V5_FEATURES

EXP76 = config.EXPERIMENTS_DIR / "exp_76_architectural_diversity_blend.json"
EXP77 = config.EXPERIMENTS_DIR / "exp_77_recursive_member_upgrade.json"

# Member B' — the Phase-5 recursive architecture plus the champion's six
# shape/cycle features. Recency and listing features stay out for the reason
# Phase 5 gave: a fractional prediction fed back in would corrupt them.
REC_COLS_V5 = list(recursive.REC_COLS) + list(V4_FEATURES) + list(V5_FEATURES)


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    t0 = time.time()
    banner("EXPERIMENT #78 — BLEND FORECAST FOR d_1942..d_1969")

    for path, label in [(EXP76, "#76"), (EXP77, "#77")]:
        if not path.exists():
            raise SystemExit(f"STOP: Experiment {label} has not been run.")
        if not json.loads(path.read_text(encoding="utf-8")).get("accepted"):
            raise SystemExit(f"STOP: Experiment {label} was not accepted — "
                             "nothing to promote.")
    rec76 = json.loads(EXP76.read_text(encoding="utf-8"))
    rec77 = json.loads(EXP77.read_text(encoding="utf-8"))
    w = float(rec77["inner_selected_weights"]["AB2"])
    print(f"  Experiment #76 accepted: {rec76['decision']}  "
          f"(4/4 windows, 3/3 seeds)")
    print(f"  Experiment #77 accepted: {rec77['decision']}  "
          f"(D1 4/4, D2 {rec77['mean_blend_dRMSE']:+.4f}, D3 "
          f"{rec77['member_wins']}/4)")
    print(f"  member A  : direct, {len(CHAMPION_FEATURES)} features")
    print(f"  member B' : recursive, {len(REC_COLS_V5)} features")
    print(f"  blend w   : {w:.2f} on member A — selected on the inner window "
          "d_1886..d_1913,")
    print(f"              which ends {config.FINAL_FORECAST_ORIGIN_IDX - 1912} "
          "days before this forecast origin")

    FO = config.FINAL_FORECAST_ORIGIN_IDX
    data = M5Data()
    fb = FeatureBuilderV5(data)
    bt = Backtester(data, feature_builder=fb)
    fw = bt.make_window(FO).describe()
    print(f"\n  forecast origin : {fw['forecast_origin_day']}")
    print(f"  forecast window : {fw['validation_days']} ({fw['validation_dates']})")

    # ==================================================================
    banner("LEAKAGE TEST AT THE FORECAST ORIGIN (direct member)")
    # ==================================================================
    clean = fb.build_origin_frame(FO, include_target=False)
    corrupt = data.sales_wide.copy()
    corrupt[:, FO + 1:] = 9999
    d2 = copy.copy(data); d2.sales_wide = corrupt
    dirty = FeatureBuilderV5(d2).build_origin_frame(FO, include_target=False)
    changed = [c for c in CHAMPION_FEATURES
               if not np.array_equal(clean[c].to_numpy(), dirty[c].to_numpy(),
                                     equal_nan=np.issubdtype(clean[c].dtype,
                                                             np.floating))]
    print(f"  {'PASS' if not changed else 'FAIL'} — {len(CHAMPION_FEATURES)} "
          f"features, {len(changed)} changed under future corruption")
    if changed:
        raise SystemExit(f"STOP: leakage in {changed}")
    del clean, dirty, corrupt, d2

    # ==================================================================
    banner("MEMBER A — direct, 38 features")
    # ==================================================================
    origins = bt.training_origins(FO, n_origins=optimize.N_ORIGINS)
    assert max(origins) + config.HORIZON <= config.LAST_KNOWN_DAY_IDX
    print(f"  training origins: {len(origins)} "
          f"(d_{origins[0]+1} .. d_{origins[-1]+1})")

    rows_per = config.N_SERIES * config.HORIZON
    X = np.empty((rows_per * len(origins), len(CHAMPION_FEATURES)), dtype=np.float32)
    Y = np.empty(rows_per * len(origins), dtype=np.float32)
    for i, o in enumerate(origins):
        f = fb.build_origin_frame(o, horizon=config.HORIZON, include_target=True)
        assert int(f["target_day_idx"].max()) <= config.LAST_KNOWN_DAY_IDX
        a, b = i * rows_per, (i + 1) * rows_per
        X[a:b] = f[CHAMPION_FEATURES].to_numpy(np.float32)
        Y[a:b] = f["sales"].to_numpy(np.float32)
        del f
    print(f"  matrix {X.shape} ({X.nbytes/1e6:.0f} MB)")
    booster_d, info_d = optimize.train(X, Y, CHAMPION_FEATURES)
    del X, Y
    mp_d = config.CHAMPION_DIR / "model_11_blend_direct_final_forecast.txt"
    booster_d.save_model(str(mp_d))
    print(f"  trained in {info_d['training_seconds']}s -> {mp_d.name}")

    future = bt.build_future_frame(FO)
    assert "sales" not in future.columns
    p_direct = np.clip(
        booster_d.predict(future[CHAMPION_FEATURES].to_numpy(np.float32)), 0, None)
    del booster_d
    print(f"  direct forecast mean {p_direct.mean():.4f}")

    # ==================================================================
    banner(f"MEMBER B' — one-step recursive, {len(REC_COLS_V5)} features")
    # ==================================================================
    booster_r, info_r = recursive.train_one_step(
        data, FO, verbose=True, builder_cls=FeatureBuilderV5, cols=REC_COLS_V5)
    mp_r = config.CHAMPION_DIR / "model_12_blend_recursive_shape_final.txt"
    booster_r.save_model(str(mp_r))
    print(f"  training origins: {info_r['training_origins_span']} "
          f"({info_r['training_origins_count']})")
    p_rec, work = recursive.recursive_forecast(
        data, booster_r, FO, builder_cls=FeatureBuilderV5, cols=REC_COLS_V5)
    checks_r = recursive.verify_no_future_leakage(data, work, FO)
    print(f"  rollout leakage checks: {checks_r}")
    if not checks_r["passed"]:
        raise SystemExit("STOP: recursive leakage check failed")
    print(f"  recursive forecast mean {p_rec.mean():.4f}")
    del booster_r, work

    # ==================================================================
    banner("BLEND")
    # ==================================================================
    preds = np.clip(w * p_direct + (1 - w) * p_rec, 0, None)
    print(f"  {len(preds):,} rows   mean {preds.mean():.4f}   "
          f"min {preds.min():.5f}   max {preds.max():.2f}")
    print(f"  correlation between the two members: "
          f"{np.corrcoef(p_direct, p_rec)[0,1]:.5f}")

    wide = preds.reshape(config.HORIZON, config.N_SERIES).T
    fc = pd.DataFrame(wide, columns=[f"F{i}" for i in range(1, config.HORIZON + 1)])
    fc.insert(0, "id", data.series_meta["id"].to_numpy())

    # ==================================================================
    banner("STRUCTURE VALIDATION")
    # ==================================================================
    sub = pd.read_csv(config.SAMPLE_SUBMISSION_CSV, usecols=["id"])
    eval_ids = sub.loc[sub["id"].str.endswith("_evaluation"), "id"]
    vals = fc.iloc[:, 1:].to_numpy()
    checks = [
        ("rows_30490", len(fc) == config.N_SERIES, f"{len(fc):,}"),
        ("cols_F1_F28", list(fc.columns[1:]) == [f"F{i}" for i in range(1, 29)],
         "F1..F28"),
        ("no_duplicate_ids", fc["id"].duplicated().sum() == 0, "0"),
        ("no_nan", not np.isnan(vals).any(), "0"),
        ("no_negative", vals.min() >= 0, f"min {vals.min():.6f}"),
        ("ids_and_order_match_template", list(fc["id"]) == list(eval_ids), "exact"),
    ]
    for nm, ok, det in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm}: {det}")
    if not all(c[1] for c in checks):
        raise SystemExit("STOP: forecast structure invalid")

    out = config.FINAL_FORECAST_DIR / "final_forecast_28day_v3_diversity_blend.csv"
    fc.to_csv(out, index=False)
    print(f"\n  wrote {out.name}")
    print("  (final_forecast_28day.csv and _v2_shape_cycle.csv are untouched)")

    prev = pd.read_csv(config.FINAL_FORECAST_DIR /
                       "final_forecast_28day_v2_shape_cycle.csv")
    corr = float(np.corrcoef(prev.iloc[:, 1:].to_numpy().ravel(), vals.ravel())[0, 1])
    print(f"\n  correlation with the shape+cycle forecast: {corr:.5f}")
    print(f"  mean shape+cycle {prev.iloc[:,1:].to_numpy().mean():.4f}  "
          f"vs blend {vals.mean():.4f}")

    exp = Experiment(
        "exp_78_blend_final_forecast",
        model_type="Blend: LightGBM direct (38 feat) + one-step recursive shape (32 feat)",
        objective="tweedie (variance_power=1.1) for both members",
        feature_set_label="diversity blend (upgraded member B') — final forecast",
        n_features=len(CHAMPION_FEATURES),
        forecast_origin_day=fw["forecast_origin_day"],
        forecast_dates=fw["validation_dates"],
        horizon=config.HORIZON, n_series=config.N_SERIES)
    exp.note("Retrain of the accepted Experiment #77 blend at forecast origin "
             "d_1941. Across the four evaluation windows this configuration "
             f"measured mean dRMSE {-0.0242:+.4f} and mean dMAE {0.0186:+.4f} "
             "against member A alone; on the primary window it scored RMSE "
             "2.0929 / MAE 1.0395.")
    exp.note("No ground truth exists for d_1942..d_1969, so no accuracy figure "
             "applies to this forecast itself. The recursive member's "
             "'future matrix equals real sales' check is vacuous at this origin "
             "for the same reason, and is reported as such.")
    exp.set(blend_weight=w,
            direct_member={"hyperparameters": info_d["params"],
                           "n_estimators": info_d["n_estimators"],
                           "training_seconds": info_d["training_seconds"],
                           "model_path": str(mp_d.relative_to(config.PROJECT_ROOT))},
            recursive_member={**{k: v for k, v in info_r.items() if k != "params"},
                              "model_path": str(mp_r.relative_to(config.PROJECT_ROOT))},
            recursive_leakage_checks=checks_r,
            prediction_path=str(out.relative_to(config.PROJECT_ROOT)),
            forecast_mean=round(float(vals.mean()), 6),
            direct_member_forecast_mean=round(float(p_direct.mean()), 6),
            recursive_member_forecast_mean=round(float(p_rec.mean()), 6),
            correlation_with_shape_cycle_forecast=round(corr, 6),
            structure_checks=[{"check": nm, "passed": bool(o)} for nm, o, _ in checks],
            validated_by="exp_76_architectural_diversity_blend + exp_77_recursive_member_upgrade")
    exp.save()
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
