"""
Deep error autopsy of the selected model (Global LightGBM + Tweedie, RMSE 2.1210).

READ-ONLY. Trains nothing, modifies no model, overwrites no existing artifact.
Everything below is computed from predictions already on disk.

The centrepiece is a set of ORACLE CEILINGS: predictors that cheat in a specific,
controlled way (they are allowed to see the validation answers) to establish how
much error is even removable. Without those, "reduce RMSE" is a wish; with them
it becomes a budget.

    python scripts/25_error_autopsy.py
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

from pipeline import config, metrics
from pipeline.data_loader import M5Data

PRED = config.PREDICTIONS_DIR / "model_04_tweedie_recency_listing_validation.csv"
OUT = config.ARTIFACTS_DIR / "error_autopsy.json"


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def main():
    t0 = time.time()
    R: dict = {}

    d = M5Data(load_prices=False)
    P = pd.read_csv(PRED).sort_values(["target_day_idx", "series_idx"]).reset_index(drop=True)
    y = P["y_true"].to_numpy(float)
    p = P["y_pred"].to_numpy(float)
    si = P["series_idx"].to_numpy()
    ti = P["target_day_idx"].to_numpy()
    hz = P["horizon"].to_numpy()
    err = p - y
    sq = err ** 2
    TOT = sq.sum()

    meta = d.series_meta
    hist = d.sales_wide[:, :config.VALIDATION_ORIGIN_IDX + 1]
    hmean = hist.mean(axis=1)

    banner("0. GLOBAL DECOMPOSITION")
    overall_mse = mse(y, p)
    bias = err.mean()
    var = err.var()
    print(f"  RMSE          = {np.sqrt(overall_mse):.4f}   MSE = {overall_mse:.4f}")
    print(f"  MSE = bias^2 + variance  ->  {bias**2:.4f} + {var:.4f}")
    print(f"  bias      = {bias:+.4f}  ({bias**2/overall_mse*100:.2f}% of MSE)")
    print(f"  variance  = {var:.4f}  ({var/overall_mse*100:.2f}% of MSE)")
    print(f"\n  Systematic offset is a rounding error. Essentially all of the MSE is")
    print(f"  variance — error that changes sign row to row, not a constant tilt.")
    R["global"] = {"RMSE": float(np.sqrt(overall_mse)), "MSE": overall_mse,
                   "bias": float(bias), "bias_sq": float(bias**2),
                   "variance": float(var),
                   "bias_share_pct": round(float(bias**2/overall_mse*100), 3)}

    # ---- under vs over ----
    under = err < 0          # predicted below actual
    R["direction"] = {
        "rows_underpredicted_pct": round(float(under.mean()*100), 2),
        "share_of_sq_error_from_underprediction_pct": round(float(sq[under].sum()/TOT*100), 2),
        "mean_shortfall_when_under": round(float((-err[under]).mean()), 4),
        "mean_excess_when_over": round(float(err[~under].mean()), 4),
    }
    print(f"\n  under-predicted rows: {R['direction']['rows_underpredicted_pct']}% "
          f"but they carry {R['direction']['share_of_sq_error_from_underprediction_pct']}% "
          f"of squared error")

    # ==================================================================
    banner("1. DEMAND-VOLUME BUCKETS (deciles of historical mean)")
    # ==================================================================
    dec = pd.qcut(hmean[si], 10, labels=False, duplicates="drop")
    rows = []
    for k in sorted(pd.unique(dec)):
        m = dec == k
        rows.append({
            "decile": int(k) + 1, "n": int(m.sum()),
            "hist_mean_range": f"{hmean[si][m].min():.2f}-{hmean[si][m].max():.2f}",
            "actual_mean": float(y[m].mean()), "pred_mean": float(p[m].mean()),
            "bias": float(err[m].mean()), "RMSE": float(np.sqrt(mse(y[m], p[m]))),
            "sq_err_share_pct": float(sq[m].sum()/TOT*100),
        })
    dv = pd.DataFrame(rows)
    print(dv.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    R["volume_deciles"] = dv.to_dict(orient="records")
    top2 = dv.tail(2)["sq_err_share_pct"].sum()
    print(f"\n  Top 2 deciles (20% of rows) carry {top2:.1f}% of squared error.")

    # ==================================================================
    banner("2. FORECAST HORIZON")
    # ==================================================================
    hrows = []
    for h in range(1, 29):
        m = hz == h
        hrows.append({"horizon": h, "RMSE": float(np.sqrt(mse(y[m], p[m]))),
                      "bias": float(err[m].mean()),
                      "actual_mean": float(y[m].mean()),
                      "sq_err_share_pct": float(sq[m].sum()/TOT*100)})
    hd = pd.DataFrame(hrows)
    R["horizon"] = hd.to_dict(orient="records")
    print(hd.head(4).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    print("   ...")
    print(hd.tail(3).to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    c = np.corrcoef(hd["horizon"], hd["RMSE"])[0, 1]
    print(f"\n  correlation(horizon, RMSE) = {c:+.3f}")
    print(f"  week 1 mean RMSE {hd[hd.horizon<=7]['RMSE'].mean():.4f} | "
          f"week 4 mean RMSE {hd[hd.horizon>21]['RMSE'].mean():.4f}")
    # is horizon RMSE explained by that day's demand level rather than distance?
    c2 = np.corrcoef(hd["actual_mean"], hd["RMSE"])[0, 1]
    print(f"  correlation(that day's actual mean, RMSE) = {c2:+.3f}")
    R["horizon_corr"] = {"with_horizon": float(c), "with_daily_demand": float(c2),
                         "week1_RMSE": float(hd[hd.horizon<=7]['RMSE'].mean()),
                         "week4_RMSE": float(hd[hd.horizon>21]['RMSE'].mean())}

    # ==================================================================
    banner("3. HIERARCHY")
    # ==================================================================
    frame = pd.DataFrame({
        "y": y, "p": p, "sq": sq, "err": err,
        "cat": meta["cat_id"].to_numpy()[si], "dept": meta["dept_id"].to_numpy()[si],
        "store": meta["store_id"].to_numpy()[si], "state": meta["state_id"].to_numpy()[si],
        "item": meta["item_id"].to_numpy()[si], "series": si,
    })
    for col in ["cat", "dept", "store", "state"]:
        g = frame.groupby(col).agg(n=("y", "size"), actual=("y", "mean"),
                                   pred=("p", "mean"), sqsum=("sq", "sum"))
        g["RMSE"] = np.sqrt(frame.groupby(col)["sq"].mean())
        g["bias"] = frame.groupby(col)["err"].mean()
        g["share_pct"] = g["sqsum"] / TOT * 100
        g = g.sort_values("share_pct", ascending=False)
        R[f"hierarchy_{col}"] = g.reset_index().to_dict(orient="records")
        print(f"\n--- by {col} ---")
        print(g[["n", "actual", "pred", "bias", "RMSE", "share_pct"]].head(7)
              .to_string(float_format=lambda v: f"{v:9.3f}"))

    # concentration
    ser = frame.groupby("series")["sq"].sum().sort_values(ascending=False)
    itm = frame.groupby("item")["sq"].sum().sort_values(ascending=False)
    conc = {
        "top_1pct_series_share_pct": float(ser.head(int(len(ser)*0.01)).sum()/TOT*100),
        "top_5pct_series_share_pct": float(ser.head(int(len(ser)*0.05)).sum()/TOT*100),
        "top_10_series_share_pct": float(ser.head(10).sum()/TOT*100),
        "top_50_items_share_pct": float(itm.head(50).sum()/TOT*100),
        "n_series_for_50pct": int((ser.cumsum()/TOT <= 0.5).sum() + 1),
    }
    R["concentration"] = conc
    print(f"\n--- concentration ---")
    print(f"  top 1% of series (305)  -> {conc['top_1pct_series_share_pct']:.1f}% of squared error")
    print(f"  top 5% of series (1524) -> {conc['top_5pct_series_share_pct']:.1f}%")
    print(f"  just {conc['n_series_for_50pct']} series of 30,490 -> 50% of all squared error")

    # ==================================================================
    banner("4. WORST INDIVIDUAL OBSERVATIONS")
    # ==================================================================
    k = 1000
    idx = np.argpartition(-sq, k)[:k]
    idx = idx[np.argsort(-sq[idx])]
    worst = pd.DataFrame({
        "series_id": meta["id"].to_numpy()[si[idx]],
        "date": pd.to_datetime(d.dates[ti[idx]]),
        "horizon": hz[idx], "actual": y[idx], "pred": p[idx],
        "sq_err": sq[idx], "hist_mean": hmean[si[idx]],
    })
    worst["ratio_actual_to_hist"] = worst["actual"] / worst["hist_mean"].clip(lower=0.01)
    print(f"  top {k} rows (0.12% of data) carry {sq[idx].sum()/TOT*100:.2f}% of squared error")
    print(f"  of those: {int((worst.actual > worst.pred).sum())} are UNDER-predictions "
          f"({(worst.actual > worst.pred).mean()*100:.1f}%)")
    print(f"  median actual on those rows: {worst.actual.median():.1f}, "
          f"median prediction: {worst.pred.median():.1f}")
    print(f"  median actual/historical-mean ratio: {worst.ratio_actual_to_hist.median():.2f}x")
    print("\n  10 worst single observations:")
    print(worst.head(10)[["series_id", "date", "actual", "pred", "hist_mean", "sq_err"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.2f}"))
    worst.head(200).to_csv(config.ARTIFACTS_DIR / "autopsy_worst_200_rows.csv", index=False)
    R["worst"] = {
        "top1000_share_pct": round(float(sq[idx].sum()/TOT*100), 2),
        "pct_underpredictions": round(float((worst.actual > worst.pred).mean()*100), 1),
        "median_actual": float(worst.actual.median()),
        "median_pred": float(worst.pred.median()),
        "median_ratio_actual_to_hist": float(worst.ratio_actual_to_hist.median()),
        "examples": worst.head(10).assign(date=worst.head(10)["date"].astype(str)
                                          ).to_dict(orient="records"),
    }

    # spike analysis
    spike = y > 2 * np.maximum(hmean[si], 0.05)
    R["spikes"] = {
        "rows_pct": round(float(spike.mean()*100), 2),
        "share_of_sq_error_pct": round(float(sq[spike].sum()/TOT*100), 2),
    }
    print(f"\n  rows where actual > 2x the series' historical mean: "
          f"{R['spikes']['rows_pct']}% of data, "
          f"{R['spikes']['share_of_sq_error_pct']}% of squared error")

    # ==================================================================
    banner("5. ORACLE CEILINGS — how much error is even removable?")
    # ==================================================================
    print("  Each of these is allowed to cheat in one specific way. They are NOT")
    print("  models; they bound what any model could achieve.\n")

    oracles = {}

    # A. perfect global rescale (already known, recomputed here)
    from scipy.optimize import minimize_scalar
    f = minimize_scalar(lambda s: mse(y, p*s), bounds=(0.5, 2.0), method="bounded").x
    oracles["A_global_rescale"] = {
        "desc": "best single multiplier applied to all predictions",
        "RMSE": float(np.sqrt(mse(y, p*f))), "param": round(float(f), 4)}

    # B. per-series optimal scale (oracle)
    dfb = pd.DataFrame({"s": si, "y": y, "p": p})
    g = dfb.groupby("s").apply(lambda t: (t.p*t.y).sum()/max((t.p*t.p).sum(), 1e-9),
                               include_groups=False)
    scale = g.reindex(range(config.N_SERIES)).fillna(1.0).to_numpy()
    pb = p * scale[si]
    oracles["B_per_series_rescale"] = {
        "desc": "optimal multiplier per series (oracle: uses the answers)",
        "RMSE": float(np.sqrt(mse(y, pb)))}

    # C. per-series oracle constant = that series' own validation mean
    cmean = dfb.groupby("s")["y"].mean().reindex(range(config.N_SERIES)).fillna(0).to_numpy()
    pc = cmean[si]
    oracles["C_per_series_oracle_mean"] = {
        "desc": "predict each series' OWN validation-window mean (oracle constant)",
        "RMSE": float(np.sqrt(mse(y, pc)))}

    # D. per-series x weekday oracle mean
    wd = d.calendar["wday"].to_numpy()[ti]
    dfd = pd.DataFrame({"s": si, "w": wd, "y": y})
    gm = dfd.groupby(["s", "w"])["y"].transform("mean").to_numpy()
    oracles["D_per_series_weekday_oracle_mean"] = {
        "desc": "predict each series' own validation mean for that weekday (oracle)",
        "RMSE": float(np.sqrt(mse(y, gm)))}

    # E. perfect predictions on the high-volume tier only
    tier_high = hmean[si] > 3.0
    pe = p.copy(); pe[tier_high] = y[tier_high]
    oracles["E_perfect_high_volume"] = {
        "desc": "our model everywhere, but PERFECT on the >3/day tier",
        "RMSE": float(np.sqrt(mse(y, pe))),
        "rows_replaced_pct": round(float(tier_high.mean()*100), 2)}

    # F. perfect on the worst 1% of rows
    pf = p.copy()
    w1 = np.argpartition(-sq, int(len(sq)*0.01))[:int(len(sq)*0.01)]
    pf[w1] = y[w1]
    oracles["F_perfect_worst_1pct_rows"] = {
        "desc": "our model everywhere, but PERFECT on the worst 1% of rows",
        "RMSE": float(np.sqrt(mse(y, pf)))}

    # G. clip predictions to a sensible max (no oracle) - sanity
    oracles["G_no_change_reference"] = {"desc": "our model as-is",
                                        "RMSE": float(np.sqrt(overall_mse))}

    od = pd.DataFrame([{"oracle": k, **v} for k, v in oracles.items()])
    od = od[["oracle", "desc", "RMSE"]].sort_values("RMSE")
    print(od.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    R["oracles"] = oracles

    base = float(np.sqrt(overall_mse))
    print(f"\n  READ THIS CAREFULLY:")
    print(f"    Our model                                     {base:.4f}")
    print(f"    Best possible per-series constant (oracle)    "
          f"{oracles['C_per_series_oracle_mean']['RMSE']:.4f}")
    print(f"    Best possible per-series x weekday (oracle)   "
          f"{oracles['D_per_series_weekday_oracle_mean']['RMSE']:.4f}")
    print(f"    Perfect on the >3/day tier (7.7% of rows)     "
          f"{oracles['E_perfect_high_volume']['RMSE']:.4f}")
    print(f"    Perfect on the worst 1% of rows               "
          f"{oracles['F_perfect_worst_1pct_rows']['RMSE']:.4f}")

    R["headroom"] = {
        "our_RMSE": base,
        "gap_to_per_series_oracle": round(base - oracles["C_per_series_oracle_mean"]["RMSE"], 4),
        "gap_to_weekday_oracle": round(base - oracles["D_per_series_weekday_oracle_mean"]["RMSE"], 4),
        "team_benchmark": 2.0324,
    }

    OUT.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {OUT.name} and autopsy_worst_200_rows.csv")
    print(f"  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
