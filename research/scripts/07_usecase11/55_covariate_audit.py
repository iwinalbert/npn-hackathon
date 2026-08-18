
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics
from pipeline.data_loader import M5Data

PRED_FILE = config.PREDICTIONS_DIR / "exp_76_diversity_blend_validation.csv"
OUT_JSON = config.ARTIFACTS_DIR / "uc11_covariate_audit.json"
REF_WEEKS = 52


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    banner("USE CASE 11 — EXTERNAL COVARIATE AUDIT (read-only)")
    data = M5Data()
    cal = data.calendar
    origin = config.FINAL_FORECAST_ORIGIN_IDX
    future = origin + 1 + np.arange(config.HORIZON)

    banner("A. AVAILABILITY AT THE REAL FORECAST ORIGIN (d_1941)")
    log(f"  forecast window: d_{future[0]+1}..d_{future[-1]+1}  "
        f"({cal['date'].iloc[future[0]].date()} .. {cal['date'].iloc[future[-1]].date()})")

    avail = {}
    for name, col in [("wday", "wday"), ("month", "month"), ("year", "year"),
                      ("event_name_1", "event_name_1"), ("event_type_1", "event_type_1"),
                      ("event_name_2", "event_name_2"), ("event_type_2", "event_type_2"),
                      ("snap_CA", "snap_CA"), ("snap_TX", "snap_TX"), ("snap_WI", "snap_WI")]:
        block = cal[col].iloc[future]
        is_event = col.startswith("event_")
        n_present = int(len(block)) if is_event else int(block.notna().sum())
        avail[name] = {"days_covered": n_present, "of": int(config.HORIZON),
                       "complete": n_present == config.HORIZON,
                       "requires_forecasting": False,
                       "source": "calendar.csv (published in advance)"}
        extra = ""
        if is_event:
            named = block.dropna().unique().tolist()
            extra = f"   events in window: {named if named else 'none'}"
        log(f"  {name:<14} {n_present}/{config.HORIZON} days{extra}")

    fweeks = np.unique(data.day_to_week[future])
    price_future = data.price_wide[:, fweeks]
    n_series_priced = int((~np.isnan(price_future)).all(axis=1).sum())
    n_series_any = int((~np.isnan(price_future)).any(axis=1).sum())
    avail["sell_price"] = {
        "future_weeks": [int(w) for w in fweeks],
        "series_with_price_all_future_weeks": n_series_priced,
        "series_with_price_some_future_week": n_series_any,
        "of": config.N_SERIES,
        "complete": n_series_priced == config.N_SERIES,
        "requires_forecasting": False,
        "source": "sell_prices.csv (covers the forecast weeks)",
    }
    log(f"  sell_price     {n_series_priced}/{config.N_SERIES} series priced in "
        f"ALL {len(fweeks)} future weeks  ({n_series_any} in at least one)")
    log(f"  promotion      0/{config.HORIZON} — NO PROMOTION FIELD EXISTS IN M5")

    banner("B. RESIDUAL STRUCTURE ON THE PRIMARY WINDOW")
    log("  If the champion's residuals are already flat across a covariate, that")
    log("  covariate has nothing left to explain and a richer encoding of it")
    log("  cannot help. Non-flat residuals are the only justification for one.")

    pred = pd.read_csv(PRED_FILE)
    y = pred["y_true"].to_numpy()
    p = pred["y_pred"].to_numpy()
    s_idx = pred["series_idx"].to_numpy()
    t_idx = pred["target_day_idx"].to_numpy()
    resid = y - p
    log(f"\n  base RMSE {metrics.rmse(y, p):.4f}   mean residual {resid.mean():+.4f}")

    v_origin = config.VALIDATION_ORIGIN_IDX
    w_origin = int(data.day_to_week[v_origin])
    tgt_week = data.day_to_week[t_idx]

    ref_block = data.price_wide[:, max(0, w_origin - REF_WEEKS + 1):w_origin + 1]
    with np.errstate(all="ignore"):
        regular = np.nanmax(ref_block.astype(np.float64), axis=1)
    price_now = data.price_wide[s_idx, tgt_week].astype(np.float64)
    reg_row = regular[s_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        discount = np.where(reg_row > 0, 1.0 - price_now / reg_row, np.nan)

    pw = data.price_wide
    same = np.ones(config.N_SERIES, dtype=np.int32)
    cur = pw[:, w_origin]
    for back in range(1, 53):
        wcol = w_origin - back
        if wcol < 0:
            break
        match = np.isclose(pw[:, wcol], cur, equal_nan=False)
        same += np.where(match & (same == back), 1, 0)
    weeks_at_price = same[s_idx]

    def table(name, groups, labels=None):
        rows = []
        for gval in np.unique(groups[~pd.isna(groups)]):
            m = groups == gval
            if m.sum() < 500:
                continue
            rows.append({
                "covariate": name,
                "bin": str(labels[gval] if labels else gval),
                "n": int(m.sum()),
                "mean_actual": float(y[m].mean()),
                "mean_pred": float(p[m].mean()),
                "mean_residual": float(resid[m].mean()),
                "RMSE": float(metrics.rmse(y[m], p[m])),
                "share_of_sq_error_pct": float((resid[m] ** 2).sum()
                                               / (resid ** 2).sum() * 100),
            })
        return rows

    all_rows = []

    bins = np.array([-np.inf, -1e-9, 0.02, 0.05, 0.10, 0.20, 0.30, np.inf])
    lab = ["price >= regular", "0-2% off", "2-5% off", "5-10% off",
           "10-20% off", "20-30% off", "30%+ off"]
    dcode = np.digitize(discount, bins[1:-1], right=True)
    dcode = np.where(np.isnan(discount), -1, dcode).astype(float)
    dcode[dcode < 0] = np.nan
    log(f"\n  --- discount vs 52-week regular price "
        f"(reference frozen at the origin) ---")
    log(f"  {'bin':<20}{'n':>10}{'actual':>9}{'pred':>9}{'resid':>9}"
        f"{'RMSE':>9}{'sqerr%':>9}")
    for r in table("discount_vs_regular", dcode, {i: l for i, l in enumerate(lab)}):
        all_rows.append(r)
        log(f"  {r['bin']:<20}{r['n']:>10,}{r['mean_actual']:>9.3f}"
            f"{r['mean_pred']:>9.3f}{r['mean_residual']:>+9.3f}"
            f"{r['RMSE']:>9.3f}{r['share_of_sq_error_pct']:>9.2f}")

    wb = np.array([0, 1, 2, 4, 8, 16, 53])
    wcode = np.digitize(weeks_at_price, wb[1:-1], right=True).astype(float)
    wlab = {i: l for i, l in enumerate(
        ["1 week", "2 weeks", "3-4 weeks", "5-8 weeks", "9-16 weeks", "17+ weeks"])}
    log(f"\n  --- weeks the current price has held (at the origin) ---")
    log(f"  {'bin':<20}{'n':>10}{'actual':>9}{'pred':>9}{'resid':>9}"
        f"{'RMSE':>9}{'sqerr%':>9}")
    for r in table("weeks_at_current_price", wcode, wlab):
        all_rows.append(r)
        log(f"  {r['bin']:<20}{r['n']:>10,}{r['mean_actual']:>9.3f}"
            f"{r['mean_pred']:>9.3f}{r['mean_residual']:>+9.3f}"
            f"{r['RMSE']:>9.3f}{r['share_of_sq_error_pct']:>9.2f}")

    ev1 = cal["event_name_1"].to_numpy()[t_idx]
    ev_flag = pd.notna(ev1).astype(float)
    log(f"\n  --- holiday / event days ---")
    log(f"  {'bin':<20}{'n':>10}{'actual':>9}{'pred':>9}{'resid':>9}"
        f"{'RMSE':>9}{'sqerr%':>9}")
    for r in table("event_day", ev_flag, {0.0: "no event", 1.0: "event day"}):
        all_rows.append(r)
        log(f"  {r['bin']:<20}{r['n']:>10,}{r['mean_actual']:>9.3f}"
            f"{r['mean_pred']:>9.3f}{r['mean_residual']:>+9.3f}"
            f"{r['RMSE']:>9.3f}{r['share_of_sq_error_pct']:>9.2f}")
    named = pd.Series(ev1).dropna().unique().tolist()
    log(f"  events present in this window: {named}")

    snap = data.snap_matrix[t_idx, data.snap_col_of_series[s_idx]].astype(float)
    log(f"\n  --- SNAP days (matched to each series' own state) ---")
    log(f"  {'bin':<20}{'n':>10}{'actual':>9}{'pred':>9}{'resid':>9}"
        f"{'RMSE':>9}{'sqerr%':>9}")
    for r in table("snap", snap, {0.0: "SNAP off", 1.0: "SNAP on"}):
        all_rows.append(r)
        log(f"  {r['bin']:<20}{r['n']:>10,}{r['mean_actual']:>9.3f}"
            f"{r['mean_pred']:>9.3f}{r['mean_residual']:>+9.3f}"
            f"{r['RMSE']:>9.3f}{r['share_of_sq_error_pct']:>9.2f}")

    banner("C. HOW MUCH IS A PERFECT DISCOUNT CORRECTION WORTH?")
    log("  Oracle: give the model the exact mean residual of every discount bin")
    log("  (an additive per-bin correction fitted on the evaluation window")
    log("  itself, so it is an upper bound that no honest model can reach).")
    for nm, codes in [("discount bin", dcode), ("weeks-at-price bin", wcode),
                      ("discount x weeks", dcode * 10 + wcode)]:
        pc = p.copy()
        valid = ~np.isnan(codes)
        cc = np.where(valid, codes, -999)
        for gv in np.unique(cc):
            m = cc == gv
            if m.sum() >= 100:
                pc[m] = pc[m] + resid[m].mean()
        r = metrics.rmse(y, np.clip(pc, 0, None))
        log(f"    oracle on {nm:<20} RMSE {r:.4f}   "
            f"({r - metrics.rmse(y, p):+.4f})")

    T = pd.DataFrame(all_rows)
    T.to_csv(config.ARTIFACTS_DIR / "uc11_covariate_residuals.csv", index=False)
    OUT_JSON.write_text(json.dumps({
        "real_forecast_origin": f"d_{origin+1}",
        "forecast_window": f"d_{future[0]+1}..d_{future[-1]+1}",
        "availability": avail,
        "residual_tables": all_rows,
        "promotion_field_exists": False,
    }, indent=2, default=str), encoding="utf-8")
    log(f"\n  wrote {OUT_JSON.name} and uc11_covariate_residuals.csv")


if __name__ == "__main__":
    main()
