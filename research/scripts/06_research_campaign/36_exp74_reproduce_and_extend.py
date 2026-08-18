
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
from pipeline.features_v2 import V2_SETS
from pipeline.features_v4 import FeatureBuilderV4, V4_FEATURES, _shrink

BASE = V2_SETS["v2_base"]
SHAPE = list(BASE) + V4_FEATURES
CHAMP_RMSE, CHAMP_MAE = 2.1210429411947650, 1.0319268155496617
EXPECTED_SHAPE_RMSE = 2.1163


class FeatureBuilderV5(FeatureBuilderV4):

    def _cycle_profiles(self, origin: int) -> dict:
        s = self.d.sales_wide
        cal = self.d.calendar
        a = max(0, origin + 1 - 728)
        blk = s[:, a:origin + 1].astype(np.float64)
        vol = blk.sum(axis=1)
        overall = blk.mean(axis=1)

        out = {}
        for col, n, name in [("month", 13, "month"), ("dom", 32, "dom")]:
            if col == "dom":
                key = pd.to_datetime(cal["date"]).dt.day.to_numpy()[a:origin + 1]
            else:
                key = cal["month"].to_numpy()[a:origin + 1]
            prof = np.ones((config.N_SERIES, n), dtype=np.float64)
            for v in range(1, n):
                m = key == v
                if m.any():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        prof[:, v] = np.where(overall > 0,
                                              blk[:, m].mean(axis=1) / overall, 1.0)
            out[name] = _shrink(prof, vol)
        return out

    def build_origin_frame(self, origin_idx, horizon=config.HORIZON,
                           series_idx=None, include_target=True):
        frame = super().build_origin_frame(origin_idx, horizon=horizon,
                                           series_idx=series_idx,
                                           include_target=include_target)
        if series_idx is None:
            series_idx = np.arange(self.d.sales_wide.shape[0])
        n_s = len(series_idx)
        td = origin_idx + 1 + np.arange(horizon)
        cal = self.d.calendar
        months = cal["month"].to_numpy()[td]
        doms = pd.to_datetime(cal["date"]).dt.day.to_numpy()[td]

        prof = self._cycle_profiles(origin_idx)
        for name, keys in [("month_ratio", months), ("dom_ratio", doms)]:
            p = prof["month" if name == "month_ratio" else "dom"]
            vals = np.empty((horizon, n_s), dtype=np.float32)
            for i, k in enumerate(keys):
                vals[i] = p[series_idx, int(k)]
            frame[name] = vals.ravel()
        return frame


V5_FEATURES = ["month_ratio", "dom_ratio"]
EXTENDED = list(SHAPE) + V5_FEATURES


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def setup(origin, builder):
    s = optimize.Setup(origin_idx=origin)
    s.fb = builder(s.data)
    s.bt = Backtester(s.data, feature_builder=s.fb)
    s.valid = s.bt.build_validation_frame(origin)
    s.y = s.valid["sales"].to_numpy()
    return s


def fit_eval(s, cols):
    X, Y = optimize.build_matrix(s, cols)
    b, info = optimize.train(X, Y, cols)
    del X, Y
    p = optimize.predict(b, s, cols)
    return metrics.rmse(s.y, p), metrics.mae(s.y, p), p, b, info


def main():
    t0 = time.time()
    R = {}

    banner("PART A — INDEPENDENT REPRODUCTION OF THE SHAPE MODEL")
    s = setup(config.VALIDATION_ORIGIN_IDX, FeatureBuilderV4)
    r, m, p, booster, info = fit_eval(s, SHAPE)
    drift = abs(r - EXPECTED_SHAPE_RMSE)
    print(f"  champion (recorded)        RMSE {CHAMP_RMSE:.4f}  MAE {CHAMP_MAE:.4f}")
    print(f"  shape model (Exp #72)      RMSE {EXPECTED_SHAPE_RMSE:.4f}")
    print(f"  shape model (reproduced)   RMSE {r:.4f}  MAE {m:.4f}")
    print(f"  reproduction drift         {drift:.2e}  "
          f"-> {'REPRODUCED' if drift < 1e-3 else 'DOES NOT REPRODUCE'}")
    print(f"  gain vs champion           dRMSE {r-CHAMP_RMSE:+.4f}  "
          f"dMAE {m-CHAMP_MAE:+.4f}")
    reproduced = drift < 1e-3
    R["reproduction"] = {"expected": EXPECTED_SHAPE_RMSE, "measured": r,
                         "drift": drift, "reproduced": bool(reproduced),
                         "MAE": m}
    if not reproduced:
        raise SystemExit("STOP: the shape result did not reproduce; not promoting.")

    mpath = config.MODELS_DIR / "exp_74_shape_champion_primary.txt"
    booster.save_model(str(mpath))
    del booster

    banner("PART B — EXTENSION: more shape axes (month, day-of-month)")
    print("  If per-series shape is the mechanism, more cyclical axes should add")
    print("  a little more. If not, the mechanism is bounded at the weekly axis.\n")

    s5 = setup(config.VALIDATION_ORIGIN_IDX, FeatureBuilderV5)
    r5, m5, _, b5, _ = fit_eval(s5, EXTENDED)
    del b5
    print(f"  shape (4 feat)     RMSE {r:.4f}  MAE {m:.4f}")
    print(f"  extended (6 feat)  RMSE {r5:.4f}  MAE {m5:.4f}   "
          f"dRMSE vs shape {r5-r:+.4f}")
    R["extension_primary"] = {"shape_RMSE": r, "extended_RMSE": r5,
                              "delta": r5 - r, "extended_MAE": m5}

    ext_better_primary = r5 < r
    ext_rows = []
    if ext_better_primary:
        print("\n  extension helps on the primary window — testing the extra windows")
        cal = s.data.calendar
        dates = pd.to_datetime(cal["date"])
        idx = lambda ds: int(cal.index[dates == pd.Timestamp(ds)][0])
        for wname, o in {"christmas_2015": idx("2015-12-25") - 14,
                         "summer_2015": idx("2015-07-15"),
                         "autumn_2015": idx("2015-10-01")}.items():
            sa = setup(o, FeatureBuilderV4)
            ra, _, _, ba, _ = fit_eval(sa, SHAPE); del ba, sa
            sb = setup(o, FeatureBuilderV5)
            rb, _, _, bb, _ = fit_eval(sb, EXTENDED); del bb, sb
            ext_rows.append({"window": wname, "shape_RMSE": ra,
                             "extended_RMSE": rb, "delta": rb - ra})
            print(f"    {wname:<18} shape {ra:.4f}   extended {rb:.4f}   "
                  f"({rb-ra:+.4f})")
        wins = sum(1 for x in ext_rows if x["delta"] < 0)
        print(f"\n  extension wins {wins}/3 extra windows")
    else:
        print("\n  extension does not help on the primary window — not pursued further")
        wins = 0
    R["extension_windows"] = ext_rows
    R["extension_wins"] = wins
    ext_accepted = bool(ext_better_primary and wins >= 2)
    R["extension_accepted"] = ext_accepted
    print(f"  extension accepted: {ext_accepted}")

    banner("DECISION")
    final_cols = EXTENDED if ext_accepted else SHAPE
    final_rmse = r5 if ext_accepted else r
    final_mae = m5 if ext_accepted else m
    label = ("shape + cycle (6 new features)" if ext_accepted
             else "shape (4 new features)")
    print(f"  NEW CHAMPION: {label}")
    print(f"    RMSE {final_rmse:.4f}  (was {CHAMP_RMSE:.4f}, "
          f"{final_rmse-CHAMP_RMSE:+.4f})")
    print(f"    MAE  {final_mae:.4f}  (was {CHAMP_MAE:.4f}, "
          f"{final_mae-CHAMP_MAE:+.4f})")
    print(f"    validated in Exp #73: 4/4 windows, 3/3 seeds")

    exp = Experiment("exp_74_shape_reproduction_and_extension",
                     model_type="LightGBM", objective="tweedie (variance_power=1.1)",
                     feature_set_label=f"NEW CHAMPION — {label}",
                     n_features=len(final_cols), features=final_cols,
                     **s.describe())
    exp.note("Experiment #74. Part A independently reproduced Experiment #72's "
             "shape-model result before any champion change, as required. Part B "
             "tested whether the shape mechanism extends to month and "
             "day-of-month axes.")
    exp.set_metrics(RMSE=final_rmse, MAE=final_mae)
    exp.set(reproduction=R["reproduction"],
            extension_primary=R["extension_primary"],
            extension_windows=ext_rows, extension_accepted=ext_accepted,
            delta_rmse_vs_old_champion=round(final_rmse - CHAMP_RMSE, 6),
            delta_mae_vs_old_champion=round(final_mae - CHAMP_MAE, 6),
            hyperparameters=info["params"], n_estimators=info["n_estimators"],
            training_seconds=info["training_seconds"],
            model_path=str(mpath.relative_to(config.PROJECT_ROOT)),
            validated_by="exp_73_shape_feature_validation (4/4 windows, 3/3 seeds)",
            decision="NEW CHAMPION")
    exp.save()

    pd.DataFrame({"series_idx": s.valid["series_idx"].to_numpy(),
                  "target_day_idx": s.valid["target_day_idx"].to_numpy(),
                  "horizon": s.valid["horizon"].to_numpy(),
                  "y_true": s.y, "y_pred": np.round(p, 5)}).to_csv(
        config.PREDICTIONS_DIR / "exp_74_new_champion_validation.csv", index=False)

    R["final"] = {"label": label, "RMSE": final_rmse, "MAE": final_mae,
                  "n_features": len(final_cols)}
    (config.ARTIFACTS_DIR / "exp74_summary.json").write_text(
        json.dumps(R, indent=2, default=str), encoding="utf-8")
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
