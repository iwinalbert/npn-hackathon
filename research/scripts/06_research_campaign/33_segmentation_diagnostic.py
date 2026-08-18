
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
PRED = config.PREDICTIONS_DIR / "model_04_tweedie_recency_listing_validation.csv"


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main():
    t0 = time.time()
    R = {}
    d = M5Data(load_prices=False)
    S = d.sales_wide
    P = pd.read_csv(PRED).sort_values(["target_day_idx", "series_idx"]).reset_index(drop=True)
    y = P.y_true.to_numpy(float)
    p = P.y_pred.to_numpy(float)
    si = P.series_idx.to_numpy()
    ti = P.target_day_idx.to_numpy()
    meta = d.series_meta

    hist = S[:, :VO + 1]
    hmean = hist.mean(axis=1)
    hzero = (hist == 0).mean(axis=1)
    hstd = hist[:, -180:].astype(np.float64).std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hcv = np.where(hmean > 0, hstd / hmean, np.nan)
    spike_rate = (hist[:, -180:] > 2 * np.maximum(hmean, 0.05)[:, None]).mean(axis=1)

    base_rmse = metrics.rmse(y, p)
    print(f"  champion on this window: RMSE {base_rmse:.6f}")

    banner("Q1 — SEGMENTATION HEADROOM (per-segment ORACLE rescale)")
    print("  For each scheme: give every segment its own optimal multiplier,")
    print("  fitted with full knowledge of the answers. That is the ceiling on")
    print("  what any segment-level specialisation could buy through scale.\n")

    def oracle_rescale(labels) -> tuple[float, int]:
        df = pd.DataFrame({"g": labels, "y": y, "p": p})
        num = df.groupby("g").apply(lambda t: (t.p * t.y).sum(), include_groups=False)
        den = df.groupby("g").apply(lambda t: (t.p * t.p).sum(), include_groups=False)
        f = (num / den.replace(0, np.nan)).fillna(1.0)
        return metrics.rmse(y, p * df["g"].map(f).to_numpy()), int(df["g"].nunique())

    def q(v, n, lab):
        return pd.qcut(v, n, labels=[f"{lab}{i+1}" for i in range(n)],
                       duplicates="drop").astype(str)

    schemes = {
        "global (single multiplier)": np.zeros(len(y), dtype=int),
        "volume decile": q(hmean[si], 10, "vol"),
        "intermittency quintile (zero%)": q(hzero[si], 5, "zero"),
        "volatility quintile (CV)": q(pd.Series(hcv).fillna(0).to_numpy()[si], 5, "cv"),
        "spike-rate quintile": q(spike_rate[si], 5, "spk"),
        "category (3)": meta.cat_id.to_numpy()[si],
        "department (7)": meta.dept_id.to_numpy()[si],
        "store (10)": meta.store_id.to_numpy()[si],
        "store x category (30)": np.char.add(
            np.char.add(meta.store_id.to_numpy()[si].astype(str), "|"),
            meta.cat_id.to_numpy()[si].astype(str)),
        "volume decile x weekday": np.char.add(
            np.char.add(q(hmean[si], 10, "vol"), "|"),
            d.calendar.weekday.to_numpy()[ti].astype(str)),
        "PER-SERIES (30,490 groups)": si.astype(str),
    }
    rows = []
    for name, lab in schemes.items():
        r, n = oracle_rescale(lab)
        rows.append({"scheme": name, "groups": n, "oracle_RMSE": r,
                     "gain_vs_champion": r - base_rmse})
        print(f"  {name:<34} {n:>6} groups   oracle RMSE {r:.4f}   "
              f"({r - base_rmse:+.4f})")
    seg = pd.DataFrame(rows)
    R["q1_segment_oracle"] = seg.to_dict("records")

    best_coarse = seg[~seg.scheme.str.startswith("PER-SERIES")].oracle_RMSE.min()
    print(f"\n  Best coarse-segment oracle: {best_coarse:.4f}  "
          f"(vs global rescale {seg.iloc[0].oracle_RMSE:.4f}, "
          f"per-series {seg.iloc[-1].oracle_RMSE:.4f})")

    banner("Q2 — WEEKDAY-SHAPE HEADROOM (recoverable from history?)")
    print("  predictor = series level at origin  x  series weekday ratio from history")
    print("  no model, no fitting, only past sales.\n")

    wd_all = d.calendar.wday.to_numpy()
    wd_t = wd_all[ti]

    level28 = hist[:, -28:].astype(np.float64).mean(axis=1)

    def weekday_ratio(weeks: int) -> np.ndarray:
        days = weeks * 7
        blk = hist[:, -days:]
        wd_blk = wd_all[VO + 1 - days: VO + 1]
        prof = np.zeros((config.N_SERIES, 8), dtype=np.float64)
        for w in range(1, 8):
            m = wd_blk == w
            prof[:, w] = blk[:, m].astype(np.float64).mean(axis=1)
        overall = blk.astype(np.float64).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(overall[:, None] > 0, prof / overall[:, None], 1.0)
        return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

    print(f"  {'predictor':<44} {'RMSE':>8} {'vs level-only':>14}")
    flat = np.tile(level28, 28).reshape(28, -1)
    pred_flat = level28[si]
    r_flat = metrics.rmse(y, pred_flat)
    print(f"  {'level only (28-day mean, repeated)':<44} {r_flat:>8.4f} {'—':>14}")
    R["q2_level_only"] = r_flat

    q2 = [{"predictor": "level only", "RMSE": r_flat}]
    for weeks in [8, 13, 26, 52]:
        ratio = weekday_ratio(weeks)
        n_obs = level28 * weeks * 7
        shrink = (n_obs / (n_obs + 20.0))[:, None]
        ratio_s = 1.0 + (ratio - 1.0) * shrink
        pred = level28[si] * ratio_s[si, wd_t]
        r = metrics.rmse(y, pred)
        q2.append({"predictor": f"level x weekday ratio ({weeks}w history)", "RMSE": r})
        print(f"  {'level x weekday ratio (' + str(weeks) + 'w history)':<44} "
              f"{r:>8.4f} {r - r_flat:>+14.4f}")

    dfo = pd.DataFrame({"s": si, "w": wd_t, "y": y})
    orc = dfo.groupby(["s", "w"])["y"].transform("mean").to_numpy()
    r_orc = metrics.rmse(y, orc)
    q2.append({"predictor": "ORACLE per-series x weekday", "RMSE": r_orc})
    print(f"  {'ORACLE per-series x weekday (from autopsy)':<44} {r_orc:>8.4f} "
          f"{r_orc - r_flat:>+14.4f}")
    R["q2"] = q2

    recovered = (r_flat - min(x["RMSE"] for x in q2[1:-1])) / (r_flat - r_orc) * 100
    print(f"\n  Share of the weekday-oracle gap recovered from history: "
          f"{recovered:.1f}%")
    R["q2_pct_of_oracle_gap_recovered"] = float(recovered)

    banner("Q2b — DOES THE CHAMPION ALREADY CAPTURE WEEKDAY SHAPE?")
    print("  If the model already knows each series' weekly shape, its residuals")
    print("  will show no weekday pattern *within* a series. If a pattern remains,")
    print("  the information is on the table.\n")

    err = p - y
    dfe = pd.DataFrame({"s": si, "w": wd_t, "e": err})
    g = dfe.groupby(["s", "w"])["e"].mean()
    sw_mean = dfe.set_index(["s", "w"]).index.map(g).to_numpy()
    ss_tot = float(np.sum((err - err.mean()) ** 2))
    ss_sw = float(np.sum((sw_mean - err.mean()) ** 2))
    print(f"  variance of residuals explained by (series x weekday) cell means: "
          f"{ss_sw / ss_tot * 100:.2f}%")
    gw = dfe.groupby("w")["e"].mean()
    w_mean = dfe["w"].map(gw).to_numpy()
    ss_w = float(np.sum((w_mean - err.mean()) ** 2))
    print(f"  ... of which plain weekday (chain-wide) explains: "
          f"{ss_w / ss_tot * 100:.2f}%")
    R["q2b_var_explained_series_weekday_pct"] = ss_sw / ss_tot * 100
    R["q2b_var_explained_weekday_only_pct"] = ss_w / ss_tot * 100

    out = config.ARTIFACTS_DIR / "segmentation_diagnostic.json"
    out.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {out.name}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
