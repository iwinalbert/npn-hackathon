
from __future__ import annotations

import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import config, metrics, optimize, recursive
from pipeline.backtest import Backtester
from pipeline.data_loader import M5Data
from pipeline.features_v2 import FeatureBuilderV2
from pipeline.features_v4 import V4_FEATURES
from pipeline.features_v5 import FeatureBuilderV5, CHAMPION_FEATURES, V5_FEATURES

OUT = Path(__file__).resolve().parent
REPRO = OUT / "reproduction"
REC_COLS_V5 = list(recursive.REC_COLS) + list(V4_FEATURES) + list(V5_FEATURES)
W = 0.60
THRESHOLD = 0.5
VO = config.VALIDATION_ORIGIN_IDX


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def occurrence(y, p, thr=THRESHOLD):
    a, q = y > 0, p >= thr
    tp = int(np.sum(a & q)); fp = int(np.sum(~a & q))
    fn = int(np.sum(a & ~q)); tn = int(np.sum(~a & ~q))
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"Accuracy": (tp + tn) / len(y), "Precision": prec, "Recall": rec,
            "F1": 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan"),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def full_metrics(y, p):
    m = metrics.evaluate(y, p)
    m.update(occurrence(y, p))
    return m


def main():
    t0 = time.time()
    R = {}
    data = M5Data()

    banner("1. LEAKAGE CORRUPTION TEST (independent, validation origin d_1913)")
    fb = FeatureBuilderV5(data)
    clean = fb.build_origin_frame(VO, include_target=False)

    corrupt = data.sales_wide.copy()
    corrupt[:, VO + 1:] = 9999
    d2 = copy.copy(data); d2.sales_wide = corrupt
    dirty = FeatureBuilderV5(d2).build_origin_frame(VO, include_target=False)
    changed = [c for c in CHAMPION_FEATURES
               if not np.array_equal(clean[c].to_numpy(), dirty[c].to_numpy(),
                                     equal_nan=np.issubdtype(clean[c].dtype, np.floating))]
    log(f"  future SALES corrupted to 9999 after d_1913")
    log(f"    features checked : {len(CHAMPION_FEATURES)}")
    log(f"    features changed : {len(changed)}  -> {'PASS' if not changed else 'FAIL ' + str(changed)}")
    del corrupt, d2, dirty
    gc.collect()

    pc = data.price_wide.copy()
    wk_of_day = data.day_to_week
    future_weeks = np.unique(wk_of_day[VO + 1:])
    pc[:, future_weeks] = 999.0
    d3 = copy.copy(data); d3.price_wide = pc
    pricey = FeatureBuilderV5(d3).build_origin_frame(VO, include_target=False)
    price_changed = [c for c in ["sell_price", "recent_avg_price",
                                 "price_rel_to_recent_avg", "price_is_missing"]
                     if not np.array_equal(clean[c].to_numpy(), pricey[c].to_numpy(),
                                           equal_nan=True)]
    log(f"  future PRICES corrupted (mirror test — these SHOULD change)")
    log(f"    price features changed : {price_changed}")
    log(f"    -> {'PASS' if 'sell_price' in price_changed else 'FAIL — future prices unused'}")
    del pc, d3, pricey
    gc.collect()

    R["leakage_test"] = {
        "origin": "d_1913",
        "features_checked": len(CHAMPION_FEATURES),
        "features_changed_under_future_sales_corruption": changed,
        "sales_corruption_passed": len(changed) == 0,
        "price_features_changed_under_future_price_corruption": price_changed,
        "price_mirror_passed": "sell_price" in price_changed,
    }

    banner("2. REPRODUCE THE SHIPPED MODEL (primary window, w=0.60)")
    bt = Backtester(data, feature_builder=fb)
    valid = bt.build_validation_frame(VO)
    y = valid["sales"].to_numpy()
    si = valid["series_idx"].to_numpy()
    hz = valid["horizon"].to_numpy()

    class S:
        pass
    s = S(); s.data = data; s.fb = fb; s.bt = bt; s.origin_idx = VO
    s.valid = valid; s.y = y
    s.origins = bt.training_origins(VO, n_origins=optimize.N_ORIGINS)

    log(f"  training origins: {len(s.origins)} (d_{s.origins[0]+1} .. d_{s.origins[-1]+1})")
    log(f"  newest training target day = d_{max(s.origins)+config.HORIZON+1}, "
        f"validation starts d_{VO+2}  -> no overlap")

    X, Y = optimize.build_matrix(s, CHAMPION_FEATURES)
    ba, ia = optimize.train(X, Y, CHAMPION_FEATURES,
                            params={"seed": 42, "bagging_seed": 42,
                                    "feature_fraction_seed": 42})
    del X, Y
    p_direct = optimize.predict(ba, s, CHAMPION_FEATURES)
    imp = pd.DataFrame({"feature": ba.feature_name(),
                        "gain": ba.feature_importance("gain")}).sort_values(
        "gain", ascending=False)
    imp["pct"] = imp.gain / imp.gain.sum() * 100
    imp.to_csv(OUT / "figures" / "champion_feature_importance.csv", index=False)
    del ba
    gc.collect()
    log(f"  member A (direct 38f)      RMSE {metrics.rmse(y,p_direct):.4f}  "
        f"MAE {metrics.mae(y,p_direct):.4f}  ({ia['training_seconds']}s)")

    br, ir = recursive.train_one_step(data, VO, seed=42,
                                      builder_cls=FeatureBuilderV5, cols=REC_COLS_V5)
    p_rec, work = recursive.recursive_forecast(data, br, VO,
                                               builder_cls=FeatureBuilderV5,
                                               cols=REC_COLS_V5)
    ck = recursive.verify_no_future_leakage(data, work, VO)
    del br, work
    gc.collect()
    log(f"  member B' (recursive 32f)  RMSE {metrics.rmse(y,p_rec):.4f}  "
        f"MAE {metrics.mae(y,p_rec):.4f}  ({ir['training_seconds']}s)")
    log(f"  rollout leakage checks: {ck}")

    p_blend = np.clip(W * p_direct + (1 - W) * p_rec, 0, None)
    rmse_b, mae_b = metrics.rmse(y, p_blend), metrics.mae(y, p_blend)

    EXPECTED = (2.0929, 1.0395)
    log(f"\n  SHIPPED BLEND w={W}")
    log(f"    reproduced : RMSE {rmse_b:.4f}  MAE {mae_b:.4f}")
    log(f"    exp_77     : RMSE {EXPECTED[0]:.4f}  MAE {EXPECTED[1]:.4f}")
    log(f"    drift      : RMSE {abs(rmse_b-EXPECTED[0]):.2e}  "
        f"MAE {abs(mae_b-EXPECTED[1]):.2e}")
    reproduced = abs(rmse_b - EXPECTED[0]) < 5e-4 and abs(mae_b - EXPECTED[1]) < 5e-4
    log(f"    -> {'REPRODUCED' if reproduced else 'DOES NOT REPRODUCE'}")
    R["reproduction"] = {"expected_RMSE": EXPECTED[0], "expected_MAE": EXPECTED[1],
                         "measured_RMSE": rmse_b, "measured_MAE": mae_b,
                         "reproduced": bool(reproduced),
                         "member_A_RMSE": metrics.rmse(y, p_direct),
                         "member_B2_RMSE": metrics.rmse(y, p_rec),
                         "recursive_leakage_checks": ck}

    pd.DataFrame({"series_idx": si, "target_day_idx": valid["target_day_idx"].to_numpy(),
                  "horizon": hz, "y_true": y,
                  "y_pred": np.round(p_blend, 5),
                  "y_pred_direct": np.round(p_direct, 5),
                  "y_pred_recursive": np.round(p_rec, 5)}).to_csv(
        REPRO / "shipped_blend_w060_validation.csv", index=False)
    log(f"  wrote {REPRO / 'shipped_blend_w060_validation.csv'}")

    banner("3. VERIFIED METRICS FOR THE SHIPPED MODEL")
    hist = data.sales_wide[:, :VO + 1].mean(axis=1)
    high = hist[si] > 3.0
    for lab, p_ in [("member A (direct 38f)", p_direct),
                    ("member B' (recursive 32f)", p_rec),
                    ("SHIPPED blend w=0.60", p_blend)]:
        m = full_metrics(y, p_)
        log(f"  {lab}")
        log(f"    RMSE {m['RMSE']:.4f}  MAE {m['MAE']:.4f}  WAPE {m['WAPE']:.4f}  "
            f"bias {m['bias']:+.4f}")
        log(f"    Acc {m['Accuracy']:.4f}  Prec {m['Precision']:.4f}  "
            f"Rec {m['Recall']:.4f}  F1 {m['F1']:.4f}")
        log(f"    high-volume RMSE {metrics.rmse(y[high], p_[high]):.4f}")
        R[lab] = {**m, "high_volume_RMSE": metrics.rmse(y[high], p_[high])}

    dec = pd.qcut(hist[si], 10, labels=False, duplicates="drop")
    sq = (p_blend - y) ** 2
    dtab = []
    for k in range(int(dec.max()) + 1):
        m_ = dec == k
        dtab.append({"decile": k + 1, "n": int(m_.sum()),
                     "actual_mean": float(y[m_].mean()),
                     "pred_mean": float(p_blend[m_].mean()),
                     "RMSE": metrics.rmse(y[m_], p_blend[m_]),
                     "MAE": metrics.mae(y[m_], p_blend[m_]),
                     "bias": metrics.bias(y[m_], p_blend[m_]),
                     "sq_err_share_pct": float(sq[m_].sum() / sq.sum() * 100)})
    pd.DataFrame(dtab).to_csv(OUT / "figures" / "decile_table.csv", index=False)

    htab = [{"horizon": h,
             "blend_RMSE": metrics.rmse(y[hz == h], p_blend[hz == h]),
             "direct_RMSE": metrics.rmse(y[hz == h], p_direct[hz == h]),
             "recursive_RMSE": metrics.rmse(y[hz == h], p_rec[hz == h]),
             "blend_MAE": metrics.mae(y[hz == h], p_blend[hz == h])}
            for h in range(1, config.HORIZON + 1)]
    pd.DataFrame(htab).to_csv(OUT / "figures" / "horizon_table.csv", index=False)

    (OUT / "audit_verification.json").write_text(
        json.dumps(R, indent=2, default=str), encoding="utf-8")
    log(f"\n  wrote audit_verification.json   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
