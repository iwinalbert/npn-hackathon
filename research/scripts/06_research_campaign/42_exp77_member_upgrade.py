
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config, metrics, optimize, recursive
from pipeline.backtest import Backtester
from pipeline.data_loader import M5Data
from pipeline.experiment import Experiment
from pipeline.features_v4 import V4_FEATURES
from pipeline.features_v5 import (FeatureBuilderV5, CHAMPION_FEATURES,
                                  V5_FEATURES, CHAMPION_RMSE, CHAMPION_MAE)

REC_COLS_V5 = list(recursive.REC_COLS) + list(V4_FEATURES) + list(V5_FEATURES)

W_FIXED = 0.5
W_GRID = np.round(np.arange(0.30, 1.001, 0.05), 2)


def log(*a):
    print(*a, flush=True)


def banner(t):
    log(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


class SharedSetup(optimize.Setup):

    def __init__(self, data, origin_idx, n_origins=optimize.N_ORIGINS):
        self.data = data
        self.fb = FeatureBuilderV5(data)
        self.bt = Backtester(data, feature_builder=self.fb)
        self.origin_idx = origin_idx
        self.window = self.bt.make_window(origin_idx).describe()
        self.valid = self.bt.build_validation_frame(origin_idx)
        self.y = self.valid["sales"].to_numpy()
        self.origins = self.bt.training_origins(origin_idx, n_origins=n_origins)
        hist = data.sales_wide[:, :origin_idx + 1].mean(axis=1)
        self.tier = pd.Series(pd.cut(
            hist[self.valid["series_idx"].to_numpy()],
            [-0.001, 0.2, 1.0, 3.0, np.inf],
            labels=["very low", "low", "medium", "high"]))
        self.high = (self.tier == "high").to_numpy()
        self.is_zero = self.y == 0


def fit_direct(s, seed=42):
    X, Y = optimize.build_matrix(s, CHAMPION_FEATURES)
    b, info = optimize.train(X, Y, CHAMPION_FEATURES,
                             params={"seed": seed, "bagging_seed": seed,
                                     "feature_fraction_seed": seed})
    del X, Y
    p = optimize.predict(b, s, CHAMPION_FEATURES)
    del b
    gc.collect()
    return p, info


def fit_recursive(s, seed=42, upgraded=False):
    from pipeline.features_v2 import FeatureBuilderV2
    cls = FeatureBuilderV5 if upgraded else FeatureBuilderV2
    cols = REC_COLS_V5 if upgraded else recursive.REC_COLS
    b, info = recursive.train_one_step(s.data, s.origin_idx, seed=seed,
                                       builder_cls=cls, cols=cols)
    p, work = recursive.recursive_forecast(s.data, b, s.origin_idx,
                                           builder_cls=cls, cols=cols)
    checks = recursive.verify_no_future_leakage(s.data, work, s.origin_idx)
    del b, work
    gc.collect()
    if not checks["passed"]:
        raise SystemExit(f"STOP: recursive leakage check failed: {checks}")
    return p, info, checks


def blend(pa, pb, w=W_FIXED):
    return np.clip(w * pa + (1.0 - w) * pb, 0, None)


def rho(pa, pb, y):
    return float(np.corrcoef(pa - y, pb - y)[0, 1])


def evaluate_window(s, name):
    log(f"\n  [{name}]  origin d_{s.origin_idx+1}  ({s.window['validation_dates']})")
    y = s.y

    pa, _ = fit_direct(s)
    log(f"    A  direct      RMSE {metrics.rmse(y,pa):.4f}  MAE {metrics.mae(y,pa):.4f}")

    pb, _, ck_b = fit_recursive(s, upgraded=False)
    log(f"    B  recursive26 RMSE {metrics.rmse(y,pb):.4f}  MAE {metrics.mae(y,pb):.4f}"
        f"   leak-ok {ck_b['passed']}")

    pb2, _, ck_b2 = fit_recursive(s, upgraded=True)
    log(f"    B' recursive32 RMSE {metrics.rmse(y,pb2):.4f}  MAE {metrics.mae(y,pb2):.4f}"
        f"   leak-ok {ck_b2['passed']}")

    bl_old = blend(pa, pb)
    bl_new = blend(pa, pb2)
    bl_3 = np.clip((pa + pb + pb2) / 3.0, 0, None)

    r = {
        "window": name, "dates": s.window["validation_dates"],
        "A_RMSE": metrics.rmse(y, pa), "A_MAE": metrics.mae(y, pa),
        "B_RMSE": metrics.rmse(y, pb), "B_MAE": metrics.mae(y, pb),
        "B2_RMSE": metrics.rmse(y, pb2), "B2_MAE": metrics.mae(y, pb2),
        "rho_A_B": rho(pa, pb, y), "rho_A_B2": rho(pa, pb2, y),
        "rho_B_B2": rho(pb, pb2, y),
        "blend_AB_RMSE": metrics.rmse(y, bl_old), "blend_AB_MAE": metrics.mae(y, bl_old),
        "blend_AB2_RMSE": metrics.rmse(y, bl_new), "blend_AB2_MAE": metrics.mae(y, bl_new),
        "blend3_RMSE": metrics.rmse(y, bl_3), "blend3_MAE": metrics.mae(y, bl_3),
        "dRMSE_upgrade": metrics.rmse(y, bl_new) - metrics.rmse(y, bl_old),
        "dMAE_upgrade": metrics.mae(y, bl_new) - metrics.mae(y, bl_old),
        "dRMSE_member": metrics.rmse(y, pb2) - metrics.rmse(y, pb),
        "A_highvol": metrics.rmse(y[s.high], pa[s.high]),
        "blend_AB_highvol": metrics.rmse(y[s.high], bl_old[s.high]),
        "blend_AB2_highvol": metrics.rmse(y[s.high], bl_new[s.high]),
        "blend3_highvol": metrics.rmse(y[s.high], bl_3[s.high]),
        "leak_B": ck_b, "leak_B2": ck_b2,
    }
    log(f"    rho(A,B) {r['rho_A_B']:.4f}   rho(A,B') {r['rho_A_B2']:.4f}   "
        f"(diversity cost {r['rho_A_B2']-r['rho_A_B']:+.4f})")
    log(f"    blend A+B   RMSE {r['blend_AB_RMSE']:.4f}  MAE {r['blend_AB_MAE']:.4f}")
    log(f"    blend A+B'  RMSE {r['blend_AB2_RMSE']:.4f}  MAE {r['blend_AB2_MAE']:.4f}"
        f"   dRMSE {r['dRMSE_upgrade']:+.4f}")
    log(f"    blend A+B+B' RMSE {r['blend3_RMSE']:.4f}  MAE {r['blend3_MAE']:.4f}")

    frontier = {}
    for tag, pb_ in [("AB", pb), ("AB2", pb2)]:
        frontier[tag] = [{"w": float(w), "RMSE": metrics.rmse(y, blend(pa, pb_, float(w))),
                          "MAE": metrics.mae(y, blend(pa, pb_, float(w)))}
                         for w in W_GRID]
    r["frontier"] = frontier
    del pa, pb, pb2, bl_old, bl_new, bl_3
    gc.collect()
    return r


def main():
    t0 = time.time()
    banner("EXPERIMENT #77 — BETTER MEMBER B, OR MORE DIFFERENT MEMBER B?")
    log(f"  member A  : direct champion, {len(CHAMPION_FEATURES)} features")
    log(f"  member B  : recursive, {len(recursive.REC_COLS)} features  (#76 configuration)")
    log(f"  member B' : recursive, {len(REC_COLS_V5)} features  (+ 6 shape/cycle)")
    log(f"  blend     : w = {W_FIXED} fixed for the acceptance test")
    log("\n  PRE-REGISTERED: D1 >=3/4 windows | D2 mean dRMSE <= -0.002 |")
    log("                  D3 member B' beats B on >=3/4 windows")

    log("\n  loading data once...")
    data = M5Data()
    cal = data.calendar
    dates = pd.to_datetime(cal["date"])
    idx = lambda ds: int(cal.index[dates == pd.Timestamp(ds)][0])

    banner("STEP 1 — INNER WINDOW (weight selection only, never an eval window)")
    log("  origin d_1885, targets d_1886..d_1913 — strictly before the primary")
    log("  window's targets and before the final forecast origin.\n")
    inner_origin = config.VALIDATION_ORIGIN_IDX - config.HORIZON
    s_in = SharedSetup(data, inner_origin)
    inner = evaluate_window(s_in, "inner_d1885")
    w_star = {}
    for tag in ("AB", "AB2"):
        fr = pd.DataFrame(inner["frontier"][tag])
        w_star[tag] = float(fr.loc[fr.RMSE.idxmin(), "w"])
    log(f"\n  inner-window RMSE-optimal weights:  A+B w={w_star['AB']:.2f}   "
        f"A+B' w={w_star['AB2']:.2f}")
    log("  these are carried to the evaluation windows as fixed constants")
    del s_in
    gc.collect()

    banner("STEP 2 — FOUR EVALUATION WINDOWS")
    WINDOWS = {
        "primary_spring_2016": config.VALIDATION_ORIGIN_IDX,
        "christmas_2015": idx("2015-12-25") - 14,
        "summer_2015": idx("2015-07-15"),
        "autumn_2015": idx("2015-10-01"),
    }
    rows = []
    for wname, o in WINDOWS.items():
        s = SharedSetup(data, o)
        rows.append(evaluate_window(s, wname))
        del s
        gc.collect()

    W = pd.DataFrame([{k: v for k, v in r.items()
                       if k not in ("frontier", "leak_B", "leak_B2")} for r in rows])

    banner("RESULTS")
    log(f"  {'window':<22}{'A':>9}{'B':>9}{'B2':>9}{'A+B':>10}{'A+B2':>10}"
        f"{'dRMSE':>9}{'rhoAB':>8}{'rhoAB2':>8}")
    for _, r in W.iterrows():
        log(f"  {r.window:<22}{r.A_RMSE:>9.4f}{r.B_RMSE:>9.4f}{r.B2_RMSE:>9.4f}"
            f"{r.blend_AB_RMSE:>10.4f}{r.blend_AB2_RMSE:>10.4f}"
            f"{r.dRMSE_upgrade:>+9.4f}{r.rho_A_B:>8.4f}{r.rho_A_B2:>8.4f}")

    wins = int((W.dRMSE_upgrade < 0).sum())
    mean_d = float(W.dRMSE_upgrade.mean())
    member_wins = int((W.dRMSE_member < 0).sum())
    mean_rho_shift = float((W.rho_A_B2 - W.rho_A_B).mean())
    log(f"\n  blend A+B' wins {wins}/4 windows   mean dRMSE {mean_d:+.4f}   "
        f"mean dMAE {W.dMAE_upgrade.mean():+.4f}")
    log(f"  member B' beats B on {member_wins}/4 windows   "
        f"mean dRMSE {W.dRMSE_member.mean():+.4f}")
    log(f"  mean diversity cost rho(A,B') - rho(A,B) = {mean_rho_shift:+.4f}")
    log(f"\n  3-way blend A+B+B' mean RMSE {W.blend3_RMSE.mean():.4f}  "
        f"vs A+B {W.blend_AB_RMSE.mean():.4f}  vs A+B' {W.blend_AB2_RMSE.mean():.4f}")
    log(f"  high-volume (mean over windows): A {W.A_highvol.mean():.4f}  "
        f"A+B {W.blend_AB_highvol.mean():.4f}  A+B' {W.blend_AB2_highvol.mean():.4f}  "
        f"3-way {W.blend3_highvol.mean():.4f}")

    banner("OPERATING POINT — inner-selected weight, applied unchanged")
    op = []
    for r in rows:
        for tag in ("AB", "AB2"):
            fr = pd.DataFrame(r["frontier"][tag])
            row = fr.loc[(fr.w - w_star[tag]).abs().idxmin()]
            base = fr.loc[fr.w == 1.0].iloc[0]
            op.append({"window": r["window"], "pair": tag, "w": float(row.w),
                       "RMSE": float(row.RMSE), "MAE": float(row.MAE),
                       "dRMSE_vs_A": float(row.RMSE - base.RMSE),
                       "dMAE_vs_A": float(row.MAE - base.MAE)})
    OP = pd.DataFrame(op)
    log(f"  {'window':<22}{'pair':>6}{'w':>6}{'RMSE':>10}{'dRMSE vs A':>12}"
        f"{'MAE':>10}{'dMAE vs A':>11}")
    for _, r in OP.iterrows():
        log(f"  {r.window:<22}{r.pair:>6}{r.w:>6.2f}{r.RMSE:>10.4f}"
            f"{r.dRMSE_vs_A:>+12.4f}{r.MAE:>10.4f}{r.dMAE_vs_A:>+11.4f}")
    for tag in ("AB", "AB2"):
        sub = OP[OP.pair == tag]
        log(f"  {tag}: mean dRMSE {sub.dRMSE_vs_A.mean():+.4f}   "
            f"mean dMAE {sub.dMAE_vs_A.mean():+.4f}   (w={w_star[tag]:.2f})")

    banner("DECISION")
    crit = {
        "D1_blend_wins_at_least_3_of_4": wins >= 3,
        "D2_mean_blend_dRMSE_at_most_-0.002": mean_d <= -0.002,
        "D3_member_B2_beats_B_on_at_least_3_of_4": member_wins >= 3,
    }
    for k, v in crit.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    accepted = all(crit.values())
    log(f"\n  -> {'MEMBER B UPGRADED' if accepted else 'MEMBER B STANDS (#76 blend unchanged)'}")

    prim = W[W.window == "primary_spring_2016"].iloc[0]
    log(f"\n  primary window summary")
    log(f"    champion (recorded, exp_74)   RMSE {CHAMPION_RMSE:.4f}  MAE {CHAMPION_MAE:.4f}")
    log(f"    A this run                    RMSE {prim.A_RMSE:.4f}  MAE {prim.A_MAE:.4f}")
    log(f"    blend A+B  (#76 config)       RMSE {prim.blend_AB_RMSE:.4f}  MAE {prim.blend_AB_MAE:.4f}")
    log(f"    blend A+B' (upgraded)         RMSE {prim.blend_AB2_RMSE:.4f}  MAE {prim.blend_AB2_MAE:.4f}")
    log(f"    blend A+B+B' (3-way)          RMSE {prim.blend3_RMSE:.4f}  MAE {prim.blend3_MAE:.4f}")

    exp = Experiment(
        "exp_77_recursive_member_upgrade",
        model_type="Blend: direct (38f) + recursive (26f vs 32f)",
        objective="tweedie (variance_power=1.1) for all members",
        feature_set_label="member B upgraded with 6 shape/cycle features",
        n_features=len(REC_COLS_V5), features=list(REC_COLS_V5),
        **SharedSetup(data, config.VALIDATION_ORIGIN_IDX).describe())
    exp.note("Experiment #77. Tests whether giving the recursive member the "
             "champion's six shape/cycle features improves the #76 blend, or "
             "merely raises the residual correlation between the members. The "
             "break-even calculation in the script header shows a 0.005 rise in "
             "correlation cancels a 0.005 RMSE gain in the member.")
    exp.note("The blend weight is held at the pre-registered 0.5 for the "
             "acceptance test so that member B's feature set is the only thing "
             "that differs. A separately-reported operating point uses a weight "
             "selected on an inner window (origin d_1885, targets d_1886..d_1913) "
             "that precedes the primary window's targets and the forecast origin.")
    exp.set_metrics(RMSE=float(prim.blend_AB2_RMSE), MAE=float(prim.blend_AB2_MAE))
    exp.set(blend_weight_acceptance_test=W_FIXED,
            inner_selected_weights=w_star,
            inner_window=  {k: v for k, v in inner.items() if k != "frontier"},
            cross_window=[{k: v for k, v in r.items() if k != "frontier"} for r in rows],
            blend_wins=wins, mean_blend_dRMSE=mean_d,
            mean_blend_dMAE=float(W.dMAE_upgrade.mean()),
            member_wins=member_wins,
            mean_member_dRMSE=float(W.dRMSE_member.mean()),
            mean_diversity_cost=mean_rho_shift,
            operating_point=op,
            acceptance_criteria=crit, accepted=accepted,
            decision=("MEMBER B UPGRADED" if accepted
                      else "MEMBER B STANDS — #76 blend unchanged"))
    exp.save()

    W.to_csv(config.ARTIFACTS_DIR / "exp77_cross_window.csv", index=False)
    OP.to_csv(config.ARTIFACTS_DIR / "exp77_operating_point.csv", index=False)
    (config.ARTIFACTS_DIR / "exp77_summary.json").write_text(json.dumps(
        {"inner": inner, "windows": rows, "w_star": w_star,
         "criteria": crit, "accepted": accepted, "blend_wins": wins,
         "mean_blend_dRMSE": mean_d, "member_wins": member_wins,
         "mean_diversity_cost": mean_rho_shift},
        indent=2, default=str), encoding="utf-8")
    log(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
