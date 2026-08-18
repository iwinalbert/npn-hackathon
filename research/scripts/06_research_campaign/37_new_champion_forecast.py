
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

sys.path.insert(0, str(config.PROJECT_ROOT / "scripts" / "06_research_campaign"))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "exp74", config.PROJECT_ROOT / "scripts" / "06_research_campaign"
    / "36_exp74_reproduce_and_extend.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
FeatureBuilderV5, EXTENDED = _m.FeatureBuilderV5, _m.EXTENDED

OLD_RMSE, OLD_MAE = 2.1210429411947650, 1.0319268155496617


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    t0 = time.time()
    banner("PROMOTING THE SHAPE+CYCLE MODEL AND FORECASTING d_1942..d_1969")
    print(f"  features: {len(EXTENDED)} (32 champion + 4 shape + 2 cycle)")

    FO = config.FINAL_FORECAST_ORIGIN_IDX
    from pipeline.data_loader import M5Data

    class _S:
        pass

    s = _S()
    s.data = M5Data()
    s.fb = FeatureBuilderV5(s.data)
    s.bt = Backtester(s.data, feature_builder=s.fb)
    s.origin_idx = FO

    fw = s.bt.make_window(FO).describe()
    print(f"  forecast origin : {fw['forecast_origin_day']} ({fw['forecast_origin_date']})")
    print(f"  forecast window : {fw['validation_days']} ({fw['validation_dates']})")

    origins = s.bt.training_origins(FO, n_origins=15)
    assert max(origins) + config.HORIZON <= config.LAST_KNOWN_DAY_IDX
    print(f"  training origins: {len(origins)} (d_{origins[0]+1} .. d_{origins[-1]+1})")

    banner("LEAKAGE TEST AT THE FORECAST ORIGIN")
    clean = s.fb.build_origin_frame(FO, include_target=False)
    corrupt = s.data.sales_wide.copy()
    corrupt[:, FO + 1:] = 9999
    d2 = copy.copy(s.data); d2.sales_wide = corrupt
    dirty = FeatureBuilderV5(d2).build_origin_frame(FO, include_target=False)
    changed = [c for c in EXTENDED
               if not np.array_equal(clean[c].to_numpy(), dirty[c].to_numpy(),
                                     equal_nan=np.issubdtype(clean[c].dtype, np.floating))]
    print(f"  {'PASS' if not changed else 'FAIL'} — {len(EXTENDED)} features, "
          f"{len(changed)} changed")
    if changed:
        raise SystemExit(f"STOP: leakage in {changed}")
    del clean, dirty, corrupt, d2

    banner("TRAIN")
    import numpy as _np
    rows_per = config.N_SERIES * config.HORIZON
    X = _np.empty((rows_per * len(origins), len(EXTENDED)), dtype=_np.float32)
    Y = _np.empty(rows_per * len(origins), dtype=_np.float32)
    for i, o in enumerate(origins):
        f = s.fb.build_origin_frame(o, horizon=config.HORIZON, include_target=True)
        assert int(f["target_day_idx"].max()) <= config.LAST_KNOWN_DAY_IDX
        a, b = i * rows_per, (i + 1) * rows_per
        X[a:b] = f[EXTENDED].to_numpy(_np.float32)
        Y[a:b] = f["sales"].to_numpy(_np.float32)
        del f
    print(f"      matrix {X.shape} ({X.nbytes/1e6:.0f} MB)")
    booster, info = optimize.train(X, Y, EXTENDED)
    del X, Y
    mpath = config.CHAMPION_DIR / "model_10_shape_cycle_final_forecast.txt"
    booster.save_model(str(mpath))
    print(f"  trained in {info['training_seconds']}s -> {mpath.name}")

    banner("FORECAST")
    future = s.bt.build_future_frame(FO)
    assert "sales" not in future.columns
    preds = optimize.predict(booster, type("S", (), {"valid": future})(), EXTENDED) \
        if False else np.clip(booster.predict(future[EXTENDED].to_numpy(np.float32)),
                              0, None)
    print(f"  {len(future):,} rows   mean {preds.mean():.4f}  "
          f"min {preds.min():.5f}  max {preds.max():.2f}")

    wide = preds.reshape(config.HORIZON, config.N_SERIES).T
    fc = pd.DataFrame(wide, columns=[f"F{i}" for i in range(1, config.HORIZON + 1)])
    fc.insert(0, "id", s.data.series_meta["id"].to_numpy())

    banner("STRUCTURE VALIDATION")
    sub = pd.read_csv(config.SAMPLE_SUBMISSION_CSV, usecols=["id"])
    eval_ids = sub.loc[sub["id"].str.endswith("_evaluation"), "id"]
    vals = fc.iloc[:, 1:].to_numpy()
    checks = [
        ("rows_30490", len(fc) == config.N_SERIES, f"{len(fc):,}"),
        ("cols_F1_F28", list(fc.columns[1:]) == [f"F{i}" for i in range(1, 29)], "F1..F28"),
        ("no_duplicate_ids", fc["id"].duplicated().sum() == 0, "0"),
        ("no_nan", not np.isnan(vals).any(), "0"),
        ("no_negative", vals.min() >= 0, f"min {vals.min():.6f}"),
        ("ids_and_order_match_template", list(fc["id"]) == list(eval_ids), "exact"),
    ]
    for n, ok, det in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}: {det}")
    if not all(c[1] for c in checks):
        raise SystemExit("STOP: forecast structure invalid")

    out = config.FINAL_FORECAST_DIR / "final_forecast_28day_v2_shape_cycle.csv"
    fc.to_csv(out, index=False)
    print(f"\n  wrote {out.name}")
    print("  (the previous forecast final_forecast_28day.csv is untouched)")

    old = pd.read_csv(config.FINAL_FORECAST_DIR / "final_forecast_28day.csv")
    corr = np.corrcoef(old.iloc[:, 1:].to_numpy().ravel(), vals.ravel())[0, 1]
    print(f"\n  correlation with previous forecast: {corr:.5f}")
    print(f"  mean previous {old.iloc[:,1:].to_numpy().mean():.4f}  "
          f"vs new {vals.mean():.4f}")

    exp = Experiment("exp_75_new_champion_final_forecast", model_type="LightGBM",
                     objective="tweedie (variance_power=1.1)",
                     feature_set_label="NEW CHAMPION shape+cycle — final forecast",
                     n_features=len(EXTENDED),
                     forecast_origin_day=fw["forecast_origin_day"],
                     forecast_dates=fw["validation_dates"],
                     horizon=config.HORIZON, n_series=config.N_SERIES)
    exp.note("Retrain of the new champion at forecast origin d_1941. Validation "
             "estimate for this configuration is RMSE 2.1157 / MAE 1.0287 "
             "(Experiment #74), validated across 4 windows and 3 seeds in #73.")
    exp.note("No ground truth exists for d_1942..d_1969, so no accuracy figure "
             "applies to this forecast itself.")
    exp.set(hyperparameters=info["params"], n_estimators=info["n_estimators"],
            training_seconds=info["training_seconds"],
            model_path=str(mpath.relative_to(config.PROJECT_ROOT)),
            prediction_path=str(out.relative_to(config.PROJECT_ROOT)),
            forecast_mean=round(float(vals.mean()), 6),
            correlation_with_previous_forecast=round(float(corr), 6),
            structure_checks=[{"check": n, "passed": bool(o)} for n, o, _ in checks])
    exp.save()
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
