
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

VO = config.VALIDATION_ORIGIN_IDX
PD_ = config.PREDICTIONS_DIR

CHAMPION_FILE = "exp_74_new_champion_validation.csv"
OLD_CHAMPION_FILE = "model_04_tweedie_recency_listing_validation.csv"

CANDIDATES = {
    "shape_36feat_tweedie1.1":      "exp_72_shape_validation.csv",
    "old_champ_32feat_tweedie1.1":  OLD_CHAMPION_FILE,
    "tweedie_1.5_32feat":           "opt_04b_power_1_5_primary_validation.csv",
    "poisson_32feat":               "opt_06_obj_poisson_validation.csv",
    "l2_32feat":                    "opt_06_obj_l2_validation.csv",
    "l1_32feat":                    "opt_06_obj_l1_validation.csv",
    "hurdle_32feat":                "model_05_hurdle_validation.csv",
    "recursive_onestep":            "opt_05_recursive_validation.csv",
    "yoy_36feat":                   "exp_71_yoy_validation.csv",
    "v2_all_46feat":                "opt_02_v2_all_validation.csv",
    "v2_demand":                    "opt_02_v2_A_demand_validation.csv",
    "v2_price":                     "opt_02_v2_C_price_validation.csv",
    "team_style":                   "model_08_team_style_validation.csv",
    "plain_lightgbm_l2":            "model_01_lightgbm_validation.csv",
}


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def load(fname: str) -> pd.DataFrame | None:
    p = PD_ / fname
    if not p.exists():
        return None
    df = pd.read_csv(p)
    need = {"series_idx", "target_day_idx", "y_true", "y_pred"}
    if not need.issubset(df.columns):
        return None
    keep = list(need | ({"horizon"} & set(df.columns)))
    return (df[keep].sort_values(["target_day_idx", "series_idx"])
            .reset_index(drop=True))


def main():
    t0 = time.time()
    R: dict = {}

    banner("LOADING")
    champ = load(CHAMPION_FILE)
    if champ is None:
        raise SystemExit(f"STOP: {CHAMPION_FILE} not found")
    y = champ.y_true.to_numpy(np.float64)
    p_champ = champ.y_pred.to_numpy(np.float64)
    si = champ.series_idx.to_numpy()
    ti = champ.target_day_idx.to_numpy()
    hz = champ.horizon.to_numpy()
    n = len(y)
    champ_rmse = metrics.rmse(y, p_champ)
    champ_mae = metrics.mae(y, p_champ)
    print(f"  champion   : {CHAMPION_FILE}")
    print(f"  rows       : {n:,}")
    print(f"  RMSE {champ_rmse:.6f}   MAE {champ_mae:.6f}   "
          f"bias {metrics.bias(y, p_champ):+.5f}")
    R["champion"] = {"file": CHAMPION_FILE, "RMSE": champ_rmse, "MAE": champ_mae,
                     "n": int(n)}

    d = M5Data(load_prices=False)
    S = d.sales_wide
    hist = S[:, :VO + 1]
    hmean = hist.mean(axis=1)
    wd_all = d.calendar.wday.to_numpy()
    wd_t = wd_all[ti]

    old = load(OLD_CHAMPION_FILE)
    p_old = old.y_pred.to_numpy(np.float64)
    if not np.array_equal(old.y_true.to_numpy(np.float64), y):
        raise SystemExit("STOP: y_true misaligned between prediction files")
    old_rmse = metrics.rmse(y, p_old)
    print(f"  old champ  : RMSE {old_rmse:.6f}   "
          f"(new champion is {champ_rmse - old_rmse:+.4f})")

    banner("Q1 — IS THE WEEKDAY-SHAPE AXIS EXHAUSTED?")
    print("  Diagnostic #33 measured, on the OLD champion, how much of the residual")
    print("  variance is explained by per-(series x weekday) cell means. If the four")
    print("  shape features did their job, that share should have dropped.\n")

    def sw_share(pred: np.ndarray) -> tuple[float, float]:
        err = pred - y
        dfe = pd.DataFrame({"s": si, "w": wd_t, "e": err})
        g = dfe.groupby(["s", "w"])["e"].mean()
        sw_mean = dfe.set_index(["s", "w"]).index.map(g).to_numpy()
        ss_tot = float(np.sum((err - err.mean()) ** 2))
        ss_sw = float(np.sum((sw_mean - err.mean()) ** 2))
        gw = dfe.groupby("w")["e"].mean()
        w_mean = dfe["w"].map(gw).to_numpy()
        ss_w = float(np.sum((w_mean - err.mean()) ** 2))
        return ss_sw / ss_tot * 100, ss_w / ss_tot * 100

    old_sw, old_w = sw_share(p_old)
    new_sw, new_w = sw_share(p_champ)
    print(f"  residual variance explained by (series x weekday) cells")
    print(f"    old champion (32 feat)  {old_sw:6.2f}%")
    print(f"    new champion (38 feat)  {new_sw:6.2f}%   "
          f"({new_sw - old_sw:+.2f} pp)")
    print(f"  ... of which chain-wide weekday alone")
    print(f"    old champion            {old_w:6.3f}%")
    print(f"    new champion            {new_w:6.3f}%")
    print()
    print("  NOTE: a (series x weekday) cell here holds only 4 observations, so a")
    print("  large share is expected from noise alone. The comparison between the")
    print("  two models is the informative part, not the level.")
    R["q1_series_weekday_residual_share"] = {
        "old_champion_pct": old_sw, "new_champion_pct": new_sw,
        "delta_pp": new_sw - old_sw,
        "old_weekday_only_pct": old_w, "new_weekday_only_pct": new_w,
    }

    def oracle_sw_correction(pred: np.ndarray) -> float:
        err = pred - y
        dfe = pd.DataFrame({"s": si, "w": wd_t, "e": err})
        g = dfe.groupby(["s", "w"])["e"].mean()
        corr = dfe.set_index(["s", "w"]).index.map(g).to_numpy()
        return metrics.rmse(y, np.clip(pred - corr, 0, None))

    orc_old = oracle_sw_correction(p_old)
    orc_new = oracle_sw_correction(p_champ)
    print(f"\n  ORACLE per-(series x weekday) additive correction")
    print(f"    old champion  {old_rmse:.4f} -> {orc_old:.4f}  ({orc_old-old_rmse:+.4f})")
    print(f"    new champion  {champ_rmse:.4f} -> {orc_new:.4f}  ({orc_new-champ_rmse:+.4f})")
    R["q1_oracle_series_weekday"] = {
        "old": {"base": old_rmse, "oracle": orc_old, "gain": orc_old - old_rmse},
        "new": {"base": champ_rmse, "oracle": orc_new, "gain": orc_new - champ_rmse},
    }

    banner("Q2 — ENSEMBLE HEADROOM ON TOP OF THE NEW CHAMPION")
    print("  For every model with predictions on disk: its own RMSE, the correlation")
    print("  of its residuals with the champion's, the best two-model blend weight")
    print("  and what that blend scores.\n")
    print("  The blend weight is chosen WITH the answers visible, so every gain")
    print("  below is an UPPER BOUND. A leakage-safe weight would be fitted on a")
    print("  pre-origin window and would do worse. Use these numbers only to decide")
    print("  whether the direction is worth a real experiment.\n")

    e_c = p_champ - y
    var_c = float(np.dot(e_c, e_c) / n)

    rows = []
    print(f"  {'model':<30}{'RMSE':>9}{'resid r':>10}{'w*':>8}"
          f"{'blend RMSE':>12}{'gain':>10}")
    print(f"  {'-'*30}{'-'*9}{'-'*10}{'-'*8}{'-'*12}{'-'*10}")
    for label, fname in CANDIDATES.items():
        df = load(fname)
        if df is None:
            print(f"  {label:<30}  (predictions not on disk — skipped)")
            continue
        if len(df) != n or not np.array_equal(df.y_true.to_numpy(np.float64), y):
            print(f"  {label:<30}  (row set differs from champion — skipped)")
            continue
        pm = df.y_pred.to_numpy(np.float64)
        e_m = pm - y
        r_resid = float(np.corrcoef(e_c, e_m)[0, 1])
        diff = e_c - e_m
        denom = float(np.dot(diff, diff))
        w = 1.0 if denom == 0 else float(-np.dot(e_m, diff) / denom)
        w_cl = float(np.clip(w, 0.0, 1.0))
        blend = np.clip(w_cl * p_champ + (1 - w_cl) * pm, 0, None)
        r_blend = metrics.rmse(y, blend)
        rows.append({
            "model": label, "file": fname, "RMSE": metrics.rmse(y, pm),
            "MAE": metrics.mae(y, pm), "resid_corr_with_champion": r_resid,
            "optimal_w_champion_unclipped": w, "optimal_w_champion": w_cl,
            "blend_RMSE": r_blend, "gain_vs_champion": r_blend - champ_rmse,
        })
        print(f"  {label:<30}{rows[-1]['RMSE']:>9.4f}{r_resid:>10.4f}"
              f"{w_cl:>8.3f}{r_blend:>12.4f}{r_blend - champ_rmse:>+10.4f}")

    E = pd.DataFrame(rows).sort_values("gain_vs_champion")
    R["q2_pairwise_blends"] = E.to_dict("records")

    best = E.iloc[0]
    print(f"\n  best single partner : {best['model']}  "
          f"(oracle-weighted gain {best['gain_vs_champion']:+.4f})")

    banner("Q2b — CEILING FOR A LINEAR STACK OF EVERYTHING ON DISK")
    mats, names = [p_champ], ["champion"]
    for row in rows:
        df = load(row["file"])
        mats.append(df.y_pred.to_numpy(np.float64))
        names.append(row["model"])
    A = np.column_stack(mats)

    w = np.full(A.shape[1], 1.0 / A.shape[1])
    AtA = A.T @ A / n
    Aty = A.T @ y / n
    for _ in range(20000):
        g = AtA @ w - Aty
        w = w - 0.5 * g / max(float(np.abs(AtA).max()), 1e-9)
        w = np.clip(w, 0, None)
        ssum = w.sum()
        if ssum > 0:
            w = w / ssum
    stack = np.clip(A @ w, 0, None)
    r_stack = metrics.rmse(y, stack)
    print(f"  simplex-constrained least-squares stack of {A.shape[1]} models")
    print(f"    RMSE {r_stack:.4f}   ({r_stack - champ_rmse:+.4f} vs champion)")
    print(f"    MAE  {metrics.mae(y, stack):.4f}")
    print("\n  non-trivial weights:")
    for nm, wi in sorted(zip(names, w), key=lambda t: -t[1]):
        if wi > 0.005:
            print(f"    {nm:<32}{wi:>8.3f}")
    R["q2b_full_stack"] = {
        "RMSE": r_stack, "MAE": metrics.mae(y, stack),
        "gain_vs_champion": r_stack - champ_rmse,
        "weights": {nm: float(wi) for nm, wi in zip(names, w) if wi > 1e-4},
        "note": "in-sample optimal weights: an upper bound, not an achievable result",
    }

    banner("Q3 — PER-SERIES RECALIBRATION HEADROOM, RE-MEASURED")
    def oracle_rescale(pred, labels):
        df = pd.DataFrame({"g": labels, "y": y, "p": pred})
        num = df.groupby("g", observed=True).apply(
            lambda t: float((t.p * t.y).sum()), include_groups=False)
        den = df.groupby("g", observed=True).apply(
            lambda t: float((t.p * t.p).sum()), include_groups=False)
        f = (num / den.replace(0, np.nan)).fillna(1.0)
        return metrics.rmse(y, pred * df["g"].map(f).to_numpy())

    per_series_old = oracle_rescale(p_old, si.astype(str))
    per_series_new = oracle_rescale(p_champ, si.astype(str))
    global_new = oracle_rescale(p_champ, np.zeros(n, dtype=int))
    print(f"  per-series ORACLE multiplier")
    print(f"    old champion  {old_rmse:.4f} -> {per_series_old:.4f}  "
          f"({per_series_old - old_rmse:+.4f})")
    print(f"    new champion  {champ_rmse:.4f} -> {per_series_new:.4f}  "
          f"({per_series_new - champ_rmse:+.4f})")
    print(f"  single global ORACLE multiplier (new champion)")
    print(f"    {champ_rmse:.4f} -> {global_new:.4f}  ({global_new - champ_rmse:+.4f})")
    R["q3_recalibration"] = {
        "per_series_oracle_old": per_series_old,
        "per_series_oracle_new": per_series_new,
        "per_series_gain_old": per_series_old - old_rmse,
        "per_series_gain_new": per_series_new - champ_rmse,
        "global_oracle_new": global_new,
        "global_gain_new": global_new - champ_rmse,
    }

    banner("Q4 — WHERE DID THE ERROR GO?")
    dec = pd.qcut(hmean[si], 10, labels=False, duplicates="drop")
    sq_old = (p_old - y) ** 2
    sq_new = (p_champ - y) ** 2
    tot_old, tot_new = sq_old.sum(), sq_new.sum()

    print(f"  {'decile':>7}{'n':>10}{'old RMSE':>11}{'new RMSE':>11}"
          f"{'dRMSE':>10}{'new bias':>10}{'sq err %':>10}")
    dec_rows = []
    for k in range(int(dec.max()) + 1):
        m = dec == k
        ro, rn = metrics.rmse(y[m], p_old[m]), metrics.rmse(y[m], p_champ[m])
        b = metrics.bias(y[m], p_champ[m])
        share = float(sq_new[m].sum() / tot_new * 100)
        dec_rows.append({"decile": k + 1, "n": int(m.sum()), "old_RMSE": ro,
                         "new_RMSE": rn, "dRMSE": rn - ro, "new_bias": b,
                         "sq_err_share_pct": share})
        print(f"  {k+1:>7}{int(m.sum()):>10,}{ro:>11.4f}{rn:>11.4f}"
              f"{rn-ro:>+10.4f}{b:>+10.4f}{share:>10.2f}")
    R["q4_volume_deciles"] = dec_rows

    under = p_champ < y
    print(f"\n  under-predicted rows            {under.mean()*100:.2f}%")
    print(f"  share of squared error from them {sq_new[under].sum()/tot_new*100:.2f}%")
    print(f"  mean shortfall when under        {(y-p_champ)[under].mean():.4f}")
    print(f"  mean excess when over            {(p_champ-y)[~under].mean():.4f}")
    R["q4_direction"] = {
        "rows_underpredicted_pct": float(under.mean() * 100),
        "share_sq_error_from_under_pct": float(sq_new[under].sum() / tot_new * 100),
        "mean_shortfall_when_under": float((y - p_champ)[under].mean()),
        "mean_excess_when_over": float((p_champ - y)[~under].mean()),
    }

    banner("Q5 — HORIZON STRUCTURE")
    hz_oracle = oracle_rescale(p_champ, hz.astype(str))
    hzw_oracle = oracle_rescale(
        p_champ, np.char.add(np.char.add(hz.astype(str), "|"), wd_t.astype(str)))
    print(f"  ORACLE per-horizon multiplier      "
          f"{champ_rmse:.4f} -> {hz_oracle:.4f}  ({hz_oracle-champ_rmse:+.4f})")
    print(f"  ORACLE per-(horizon x weekday)     "
          f"{champ_rmse:.4f} -> {hzw_oracle:.4f}  ({hzw_oracle-champ_rmse:+.4f})")

    hz_rows = []
    print(f"\n  {'h':>4}{'old RMSE':>11}{'new RMSE':>11}{'dRMSE':>10}{'new bias':>11}")
    for h in range(1, config.HORIZON + 1):
        m = hz == h
        ro, rn = metrics.rmse(y[m], p_old[m]), metrics.rmse(y[m], p_champ[m])
        hz_rows.append({"horizon": h, "old_RMSE": ro, "new_RMSE": rn,
                        "dRMSE": rn - ro, "new_bias": metrics.bias(y[m], p_champ[m])})
        print(f"  {h:>4}{ro:>11.4f}{rn:>11.4f}{rn-ro:>+10.4f}"
              f"{metrics.bias(y[m], p_champ[m]):>+11.4f}")
    w1 = np.polyfit(np.arange(1, 29), [r["new_RMSE"] for r in hz_rows], 1)[0]
    print(f"\n  linear trend of RMSE with horizon: {w1:+.5f} per day")
    R["q5_horizon"] = {"oracle_per_horizon": hz_oracle,
                       "oracle_per_horizon_gain": hz_oracle - champ_rmse,
                       "oracle_horizon_x_weekday": hzw_oracle,
                       "oracle_horizon_x_weekday_gain": hzw_oracle - champ_rmse,
                       "rmse_slope_per_day": float(w1), "by_horizon": hz_rows}

    out = config.ARTIFACTS_DIR / "exp76_headroom_diagnostic.json"
    out.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    E.to_csv(config.ARTIFACTS_DIR / "exp76_pairwise_blends.csv", index=False)
    print(f"\n  wrote {out.name} and exp76_pairwise_blends.csv   "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
