
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, feature_sets, metrics, models
from pipeline.backtest import Backtester
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment

POWER = 1.5
FEATURE_SET = "base_recency_listing"


def main():
    t0 = time.time()
    probe = pd.read_csv(config.ARTIFACTS_DIR / "tweedie_power_probe.csv")
    best = probe.sort_values("inner_RMSE").iloc[0]
    assert abs(float(best["power"]) - POWER) < 1e-9, (
        f"inner window selected power {best['power']}, not {POWER} — "
        "refusing to apply a setting that was not the measured winner"
    )

    print("=" * 78)
    print(f"APPLYING TWEEDIE POWER {POWER} TO THE PRIMARY WINDOW (single run)")
    print("=" * 78)
    print(f"  chosen on inner window d_1886..d_1913: RMSE {best['inner_RMSE']:.4f}")
    print(f"  (current setting 1.1 scored "
          f"{probe[probe.power == 1.1].iloc[0]['inner_RMSE']:.4f} there)\n")

    data = M5Data()
    bt = Backtester(data)
    VO = config.VALIDATION_ORIGIN_IDX
    wd = bt.make_window(VO).describe()
    valid = bt.build_validation_frame(VO)
    y = valid["sales"].to_numpy()
    cols = feature_sets.get(FEATURE_SET)
    origins = bt.training_origins(VO, n_origins=15)

    exp = Experiment(
        "model_09_tweedie_power_1_5",
        model_type="LightGBM",
        objective=f"tweedie (variance_power={POWER})",
        feature_set=FEATURE_SET,
        feature_groups=feature_sets.groups_in(FEATURE_SET),
        n_features=len(cols),
        validation_origin_day=wd["forecast_origin_day"],
        validation_days=wd["validation_days"],
        validation_dates=wd["validation_dates"],
        horizon=config.HORIZON, n_series=config.N_SERIES,
        validation_rows=int(len(valid)),
        selection_basis="tweedie_variance_power chosen on the inner window "
                        "d_1886..d_1913 only; primary window unused in selection",
    )
    exp.note("Everything except tweedie_variance_power is identical to Model 4, "
             "so any difference is attributable to that one parameter.")

    X, yt, binfo = models.build_training_matrix(
        bt, origins, cols, validation_origin=VO, verbose=False)
    exp.set(training_data=binfo, training_rows=binfo["rows"],
            training_origins=binfo["origin_days"])

    booster, minfo = models.train_lightgbm(
        X, yt, cols,
        params={"objective": "tweedie", "tweedie_variance_power": POWER,
                "metric": "rmse"},
        n_estimators=400, verbose=True)
    exp.set(**{k: v for k, v in minfo.items() if k != "params"})
    exp.set(hyperparameters=minfo["params"])
    del X, yt

    mpath = config.MODELS_DIR / "model_09_tweedie_power_1_5.txt"
    booster.save_model(str(mpath))
    exp.set(model_path=str(mpath.relative_to(config.PROJECT_ROOT)))

    preds = models.predict_nonneg(booster, valid[cols].to_numpy(np.float32))
    if np.isnan(preds).any():
        exp.error("NaN predictions"); exp.save(); raise SystemExit("STOP: NaN")

    m = metrics.evaluate(y, preds)
    exp.set_metrics(**m)

    hist = data.sales_wide[:, :VO + 1].mean(axis=1)
    tier = pd.Series(pd.cut(hist[valid["series_idx"].to_numpy()],
                            [-0.001, 0.2, 1.0, 3.0, np.inf],
                            labels=["very low", "low", "medium", "high"]))
    hm = metrics.evaluate(y[(tier == "high").to_numpy()],
                          preds[(tier == "high").to_numpy()])
    exp.set(high_volume_RMSE=hm["RMSE"], high_volume_bias=hm["bias"])

    pdf = pd.DataFrame({
        "series_idx": valid["series_idx"].to_numpy(),
        "target_day_idx": valid["target_day_idx"].to_numpy(),
        "horizon": valid["horizon"].to_numpy(),
        "y_true": y, "y_pred": np.round(preds, 5),
    })
    ppath = config.PREDICTIONS_DIR / "model_09_tweedie_power_1_5_validation.csv"
    pdf.to_csv(ppath, index=False)
    exp.set(prediction_path=str(ppath.relative_to(config.PROJECT_ROOT)))
    exp.save()

    print("\n" + "=" * 78)
    print("PRIMARY WINDOW RESULT (d_1914..d_1941, 853,720 predictions)")
    print("=" * 78)
    print(f"  Model 4  (power 1.1) : RMSE 2.1210  MAE 1.0319")
    print(f"  Model 9  (power 1.5) : RMSE {m['RMSE']:.4f}  MAE {m['MAE']:.4f}  "
          f"WAPE {m['WAPE']:.4f}  bias {m['bias']:+.4f}")
    print(f"  change               : RMSE {m['RMSE'] - 2.1210:+.4f}  "
          f"MAE {m['MAE'] - 1.0319:+.4f}")
    print(f"\n  high-volume tier     : RMSE {hm['RMSE']:.4f}  bias {hm['bias']:+.4f}"
          f"   (was 5.9756 / -0.389)")
    print(f"\n  team reported        : RMSE 2.0324  MAE 1.0869 "
          f"(their setup, not verified)")
    print(f"\n  total wall time: {time.time() - t0:.0f}s")

    out = {"model_9_power_1_5": m, "model_4_power_1_1": {"RMSE": 2.1210, "MAE": 1.0319},
           "high_volume": hm, "selected_on": "inner window d_1886..d_1913"}
    (config.ARTIFACTS_DIR / "tweedie_power_applied.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
