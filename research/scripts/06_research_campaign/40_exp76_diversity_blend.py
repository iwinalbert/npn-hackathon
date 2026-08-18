"""
EXPERIMENT #76 — ARCHITECTURAL-DIVERSITY BLEND (direct + recursive).

THE HYPOTHESIS
--------------
The champion's remaining error is almost entirely variance, not bias: the error
autopsy put the bias share of MSE at 0.11%. Variance is what averaging removes.

Experiment #70 tried averaging and failed (2.1210 -> 2.1261, REJECTED). But it
ensembled six DIRECT LightGBM models that differed only in Tweedie power, leaf
count and seed. Their residuals correlated 0.9897, and five of the six were
individually 0.010-0.038 RMSE worse than the champion. That experiment therefore
tested "do near-copies of the champion help", not "does averaging help". Its
failure is fully explained by member similarity and member quality, both of which
are fixable.

Diagnostic #76-D re-measured every model whose predictions are on disk against
the NEW 38-feature champion. One stands out:

    model                    RMSE     residual corr with champion
    champion (direct, 38f) 2.1163                   1.000
    recursive one-step     2.1182                   0.9529   <--
    every other model      2.121-2.243              0.958-0.990

The recursive model is a structurally different forecaster - one model that only
ever predicts one day ahead, rolled forward 28 times on its own output - and it
is the only model on disk that is both as accurate as the champion and materially
decorrelated from it.

For two forecasters with equal MSE sigma^2 and residual correlation rho, the
equal-weight blend has MSE = sigma^2 (1 + rho) / 2. With sigma^2 = 4.4788 and
rho = 0.9529 that predicts RMSE 2.0904. Measured on the saved predictions:
2.0912. The arithmetic and the data agree, which is why this is worth a real
experiment rather than another feature.

THE COMBINATION RULE IS FIXED IN ADVANCE, NOT FITTED
----------------------------------------------------
    blend = 0.5 * champion + 0.5 * recursive

w = 0.5 is chosen a priori, not tuned: for two members of near-identical MSE
(2.1163 vs 2.1182) the variance-minimising weight is exactly 0.5 analytically.
No parameter of this model is selected using validation data at any point. The
full w-grid is reported per window as evidence, but it never feeds back into the
configuration.

PRE-REGISTERED ACCEPTANCE CRITERIA  (fixed before this script was first run)
---------------------------------------------------------------------------
NEW CHAMPION requires ALL FOUR:
  C1  blend beats the champion on RMSE in at least 3 of 4 windows
  C2  mean window dRMSE <= -0.010
        Stricter than #73's -0.003 on purpose. The free diagnostic predicts
        -0.025 on the primary window; requiring at least 40% of that to survive
        on unseen windows guards against the primary window being optimistic.
  C3  blend beats the champion on at least 2 of 3 seeds on the primary window,
      with BOTH members reseeded
  C4  MECHANISM: mean member residual correlation across the 4 windows <= 0.97
        This is the falsifiable part. The claim is that the gain comes from
        architectural diversity. If the two architectures turn out to be ~0.99
        correlated once retrained on other windows, the stated mechanism is
        wrong and the result should not be trusted even if C1-C3 pass.

NEGATIVE CONTROL (reported, not a gate)
  champion(seed 42) blended with champion(seed 7) - same architecture, different
  seed. If plain averaging were the whole story this control would gain as much
  as the diversity blend. The gap between them is the part attributable to
  architecture rather than to averaging.

DISCLOSED COST (reported, deliberately NOT a gate)
  The recursive member has MAE 1.0717 against the champion's 1.0287, so the
  RMSE-optimal blend costs about +0.015 MAE. RMSE is this project's headline
  metric and the criteria above are set on RMSE alone, but the trade-off is real
  and the full w-frontier is reported so it is visible rather than buried. A
  near-MAE-neutral point on that frontier is reported alongside.

    python scripts/06_research_campaign/40_exp76_diversity_blend.py
"""

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
from pipeline.features_v5 import (FeatureBuilderV5, CHAMPION_FEATURES,
                                  CHAMPION_RMSE, CHAMPION_MAE)

W_BLEND = 0.5                      # fixed a priori, never fitted
SEEDS = [42, 7, 202]
W_GRID = np.round(np.arange(0.30, 1.001, 0.05), 2)


# --------------------------------------------------------------------------
class SharedSetup(optimize.Setup):
    """optimize.Setup, but reusing one already-loaded M5Data across windows."""

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


def banner(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def fit_direct(s, seed):
    """Train the champion (direct, 38 features) and predict the window."""
    X, Y = optimize.build_matrix(s, CHAMPION_FEATURES)
    b, info = optimize.train(X, Y, CHAMPION_FEATURES,
                             params={"seed": seed, "bagging_seed": seed,
                                     "feature_fraction_seed": seed})
    del X, Y
    p = optimize.predict(b, s, CHAMPION_FEATURES)
    del b
    gc.collect()
    return p, info


def fit_recursive(s, seed):
    """Train the one-step model and roll it forward across the window."""
    b, info = recursive.train_one_step(s.data, s.origin_idx, seed=seed)
    p, work = recursive.recursive_forecast(s.data, b, s.origin_idx)
    checks = recursive.verify_no_future_leakage(s.data, work, s.origin_idx)
    del b, work
    gc.collect()
    if not checks["passed"]:
        raise SystemExit(f"STOP: recursive leakage check failed: {checks}")
    return p, info, checks


def blend(pc, pr, w=W_BLEND):
    return np.clip(w * pc + (1.0 - w) * pr, 0, None)


def resid_corr(pc, pr, y):
    return float(np.corrcoef(pc - y, pr - y)[0, 1])


def w_frontier(pc, pr, y):
    rows = []
    for w in W_GRID:
        b = blend(pc, pr, float(w))
        rows.append({"w": float(w), "RMSE": metrics.rmse(y, b),
                     "MAE": metrics.mae(y, b)})
    return rows


# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    R: dict = {"pre_registered": {
        "w_blend": W_BLEND,
        "w_chosen_by": "fixed a priori (equal-MSE members); never fitted on data",
        "C1_wins_at_least_3_of_4_windows": None,
        "C2_mean_window_dRMSE_at_most_-0.010": None,
        "C3_wins_at_least_2_of_3_seeds": None,
        "C4_mean_member_residual_corr_at_most_0.97": None,
    }}

    banner("EXPERIMENT #76 — ARCHITECTURAL-DIVERSITY BLEND (direct + recursive)")
    print("  member A : champion, DIRECT, 38 features, Tweedie(1.1), 15 origins x 28d")
    print("  member B : one-step model, 26 features, 420 daily origins,")
    print("             rolled forward 28 days on its own output")
    print(f"  blend    : {W_BLEND:.2f} * A + {1-W_BLEND:.2f} * B   (fixed a priori)")
    print("\n  PRE-REGISTERED: C1 >=3/4 windows | C2 mean dRMSE <= -0.010 |")
    print("                  C3 >=2/3 seeds   | C4 mean resid corr <= 0.97")

    print("\n  loading data once and sharing it across every window...")
    data = M5Data()

    cal = data.calendar
    dates = pd.to_datetime(cal["date"])
    idx = lambda ds: int(cal.index[dates == pd.Timestamp(ds)][0])
    WINDOWS = {
        "primary_spring_2016": config.VALIDATION_ORIGIN_IDX,
        "christmas_2015": idx("2015-12-25") - 14,
        "summer_2015": idx("2015-07-15"),
        "autumn_2015": idx("2015-10-01"),
    }

    # ==================================================================
    banner("TEST A — CROSS-WINDOW (both members retrained from scratch per window)")
    # ==================================================================
    rows, frontiers, leak = [], {}, {}
    primary_preds = None
    for wname, o in WINDOWS.items():
        print(f"\n  [{wname}]  origin d_{o+1}")
        s = SharedSetup(data, o)
        print(f"    window {s.window['validation_dates']}")

        pc, info_d = fit_direct(s, 42)
        rc, mc = metrics.rmse(s.y, pc), metrics.mae(s.y, pc)
        print(f"    direct     RMSE {rc:.4f}  MAE {mc:.4f}  ({info_d['training_seconds']}s)")

        pr, info_r, checks = fit_recursive(s, 42)
        rr, mr = metrics.rmse(s.y, pr), metrics.mae(s.y, pr)
        print(f"    recursive  RMSE {rr:.4f}  MAE {mr:.4f}  ({info_r['training_seconds']}s)")
        print(f"    leakage checks: {checks}")
        leak[wname] = checks

        rho = resid_corr(pc, pr, s.y)
        b = blend(pc, pr)
        rb, mb = metrics.rmse(s.y, b), metrics.mae(s.y, b)

        # variance-cancellation prediction from the two member MSEs and rho
        ec, er = pc - s.y, pr - s.y
        pred_mse = float(np.mean(((ec + er) / 2) ** 2))
        hv = metrics.rmse(s.y[s.high], b[s.high])
        hv_c = metrics.rmse(s.y[s.high], pc[s.high])

        rows.append({
            "window": wname, "dates": s.window["validation_dates"],
            "direct_RMSE": rc, "direct_MAE": mc,
            "recursive_RMSE": rr, "recursive_MAE": mr,
            "resid_corr": rho, "blend_RMSE": rb, "blend_MAE": mb,
            "dRMSE": rb - rc, "dMAE": mb - mc,
            "blend_highvol_RMSE": hv, "direct_highvol_RMSE": hv_c,
            "highvol_dRMSE": hv - hv_c,
            "predicted_blend_RMSE": float(np.sqrt(pred_mse)),
        })
        print(f"    resid corr {rho:.4f}")
        print(f"    BLEND      RMSE {rb:.4f}  MAE {mb:.4f}   "
              f"dRMSE {rb-rc:+.4f}  dMAE {mb-mc:+.4f}")
        print(f"    high-volume RMSE {hv_c:.4f} -> {hv:.4f}  ({hv-hv_c:+.4f})")

        frontiers[wname] = w_frontier(pc, pr, s.y)
        if wname == "primary_spring_2016":
            primary_preds = {"s": s, "pc": pc.copy(), "pr": pr.copy(),
                             "info_d": info_d, "info_r": info_r}
        else:
            del s
        del pc, pr, b
        gc.collect()

    W = pd.DataFrame(rows)
    wins = int((W.dRMSE < 0).sum())
    mean_d = float(W.dRMSE.mean())
    mean_dm = float(W.dMAE.mean())
    mean_rho = float(W.resid_corr.mean())
    print(f"\n  blend wins {wins}/4 windows   mean dRMSE {mean_d:+.4f}   "
          f"mean dMAE {mean_dm:+.4f}")
    print(f"  mean member residual correlation {mean_rho:.4f}")
    print(f"  probability of {wins}/4 by chance if the blend were useless: "
          f"{[1,4,6,4,1][wins]/16:.3f}")

    # ==================================================================
    banner("TEST B — SEED SENSITIVITY (primary window, both members reseeded)")
    # ==================================================================
    s = primary_preds["s"]
    seed_rows = []
    seed_direct = {42: primary_preds["pc"]}
    pc42, pr42 = primary_preds["pc"], primary_preds["pr"]
    seed_rows.append({
        "seed": 42, "direct_RMSE": metrics.rmse(s.y, pc42),
        "recursive_RMSE": metrics.rmse(s.y, pr42),
        "blend_RMSE": metrics.rmse(s.y, blend(pc42, pr42)),
        "dRMSE": metrics.rmse(s.y, blend(pc42, pr42)) - metrics.rmse(s.y, pc42),
        "dMAE": metrics.mae(s.y, blend(pc42, pr42)) - metrics.mae(s.y, pc42),
        "resid_corr": resid_corr(pc42, pr42, s.y)})
    print(f"  seed 42   direct {seed_rows[0]['direct_RMSE']:.4f}   "
          f"blend {seed_rows[0]['blend_RMSE']:.4f}   "
          f"dRMSE {seed_rows[0]['dRMSE']:+.4f}")

    for seed in SEEDS[1:]:
        pc_s, _ = fit_direct(s, seed)
        pr_s, _, ck = fit_recursive(s, seed)
        if not ck["passed"]:
            raise SystemExit("STOP: leakage check failed on reseeded recursive")
        b_s = blend(pc_s, pr_s)
        seed_direct[seed] = pc_s.copy()
        seed_rows.append({
            "seed": seed, "direct_RMSE": metrics.rmse(s.y, pc_s),
            "recursive_RMSE": metrics.rmse(s.y, pr_s),
            "blend_RMSE": metrics.rmse(s.y, b_s),
            "dRMSE": metrics.rmse(s.y, b_s) - metrics.rmse(s.y, pc_s),
            "dMAE": metrics.mae(s.y, b_s) - metrics.mae(s.y, pc_s),
            "resid_corr": resid_corr(pc_s, pr_s, s.y)})
        print(f"  seed {seed:<4} direct {seed_rows[-1]['direct_RMSE']:.4f}   "
              f"blend {seed_rows[-1]['blend_RMSE']:.4f}   "
              f"dRMSE {seed_rows[-1]['dRMSE']:+.4f}")
        del pc_s, pr_s, b_s
        gc.collect()

    S = pd.DataFrame(seed_rows)
    seed_wins = int((S.dRMSE < 0).sum())
    print(f"\n  blend wins {seed_wins}/{len(SEEDS)} seeds   "
          f"mean dRMSE {S.dRMSE.mean():+.4f}")

    # ==================================================================
    banner("NEGATIVE CONTROL — same architecture, different seed")
    # ==================================================================
    print("  If plain averaging were the whole story, blending the champion with")
    print("  a reseeded champion would gain as much as blending it with a")
    print("  different architecture. This measures the difference.\n")
    ctrl = blend(seed_direct[42], seed_direct[7])
    r_ctrl = metrics.rmse(s.y, ctrl)
    rho_ctrl = resid_corr(seed_direct[42], seed_direct[7], s.y)
    r_d42 = metrics.rmse(s.y, seed_direct[42])
    r_div = metrics.rmse(s.y, blend(pc42, pr42))
    print(f"  champion(42)                       RMSE {r_d42:.4f}")
    print(f"  champion(42) + champion(7)         RMSE {r_ctrl:.4f}  "
          f"({r_ctrl - r_d42:+.4f})   resid corr {rho_ctrl:.4f}")
    print(f"  champion(42) + recursive(42)       RMSE {r_div:.4f}  "
          f"({r_div - r_d42:+.4f})   resid corr {resid_corr(pc42, pr42, s.y):.4f}")
    same_arch_gain = r_ctrl - r_d42
    div_gain = r_div - r_d42
    print(f"\n  attributable to averaging alone   {same_arch_gain:+.4f}")
    print(f"  attributable to architecture      {div_gain - same_arch_gain:+.4f}")
    R["negative_control"] = {
        "champion_seed42_RMSE": r_d42,
        "same_architecture_blend_RMSE": r_ctrl,
        "same_architecture_gain": same_arch_gain,
        "same_architecture_resid_corr": rho_ctrl,
        "diversity_blend_RMSE": r_div,
        "diversity_gain": div_gain,
        "gain_attributable_to_architecture": div_gain - same_arch_gain,
    }

    # ==================================================================
    banner("DISCLOSED COST — the RMSE / MAE frontier on the primary window")
    # ==================================================================
    fr = pd.DataFrame(frontiers["primary_spring_2016"])
    base_r = metrics.rmse(s.y, pc42)
    base_m = metrics.mae(s.y, pc42)
    print(f"  {'w':>6}{'RMSE':>10}{'dRMSE':>10}{'MAE':>10}{'dMAE':>10}")
    for _, r in fr.iterrows():
        print(f"  {r.w:>6.2f}{r.RMSE:>10.4f}{r.RMSE-base_r:>+10.4f}"
              f"{r.MAE:>10.4f}{r.MAE-base_m:>+10.4f}")
    mae_neutral = fr[fr.MAE - base_m <= 0.002]
    if len(mae_neutral):
        mn = mae_neutral.loc[mae_neutral.RMSE.idxmin()]
        print(f"\n  near-MAE-neutral point: w={mn.w:.2f}  RMSE {mn.RMSE:.4f} "
              f"({mn.RMSE-base_r:+.4f})  MAE {mn.MAE:.4f} ({mn.MAE-base_m:+.4f})")
        R["mae_neutral_point"] = mn.to_dict()

    # ==================================================================
    banner("DECISION")
    # ==================================================================
    crit = {
        "C1_wins_at_least_3_of_4_windows": wins >= 3,
        "C2_mean_window_dRMSE_at_most_-0.010": mean_d <= -0.010,
        "C3_wins_at_least_2_of_3_seeds": seed_wins >= 2,
        "C4_mean_member_residual_corr_at_most_0.97": mean_rho <= 0.97,
    }
    for k, v in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    accepted = all(crit.values())
    print(f"\n  -> {'NEW CHAMPION' if accepted else 'CHAMPION STANDS'}")

    b_primary = blend(pc42, pr42)
    d = optimize.diagnostics(s.y, b_primary, s)
    print(f"\n  blend on the primary window:")
    print(f"    RMSE {d['RMSE']:.4f}  (champion {CHAMPION_RMSE:.4f}, "
          f"{d['RMSE']-CHAMPION_RMSE:+.4f})")
    print(f"    MAE  {d['MAE']:.4f}  (champion {CHAMPION_MAE:.4f}, "
          f"{d['MAE']-CHAMPION_MAE:+.4f})")
    print(f"    high-volume RMSE {d['high_volume_RMSE']:.4f}")

    exp = Experiment(
        "exp_76_architectural_diversity_blend",
        model_type="Blend: LightGBM direct (38 feat) + LightGBM one-step recursive (26 feat)",
        objective="tweedie (variance_power=1.1) for both members",
        feature_set_label="champion 38 features + recursive 26 features, 0.5/0.5",
        n_features=len(CHAMPION_FEATURES), features=list(CHAMPION_FEATURES),
        **s.describe())
    exp.note("Experiment #76. Diagnostic #76-D found the recursive one-step model "
             "is the only model on disk that is as accurate as the champion "
             "(2.1182 vs 2.1163) while being materially decorrelated from it "
             "(residual corr 0.9529 against 0.958-0.990 for everything else). "
             "Experiment #70's ensemble failed because its six members were all "
             "direct models correlating 0.9897 and were individually worse; this "
             "corrects both faults.")
    exp.note("The blend weight is fixed at 0.5 a priori because the two members "
             "have near-identical MSE, for which 0.5 is the variance-minimising "
             "weight analytically. No parameter is selected using validation data.")
    exp.note("Acceptance criteria C1-C4 were written into the script header "
             "before it was first run.")
    exp.set_metrics(**d)
    exp.set(blend_weight=W_BLEND,
            cross_window=rows, window_wins=wins,
            mean_window_dRMSE=mean_d, mean_window_dMAE=mean_dm,
            mean_member_residual_corr=mean_rho,
            seed_runs=seed_rows, seed_wins=seed_wins,
            negative_control=R["negative_control"],
            w_frontier_primary=frontiers["primary_spring_2016"],
            leakage_checks=leak,
            direct_member_hyperparameters=primary_preds["info_d"]["params"],
            recursive_member=primary_preds["info_r"],
            delta_rmse_vs_champion=round(d["RMSE"] - CHAMPION_RMSE, 6),
            delta_mae_vs_champion=round(d["MAE"] - CHAMPION_MAE, 6),
            acceptance_criteria=crit, accepted=accepted,
            decision="NEW CHAMPION" if accepted else "CHAMPION STANDS")
    exp.save()

    pd.DataFrame({
        "series_idx": s.valid["series_idx"].to_numpy(),
        "target_day_idx": s.valid["target_day_idx"].to_numpy(),
        "horizon": s.valid["horizon"].to_numpy(),
        "y_true": s.y, "y_pred": np.round(b_primary, 5),
    }).to_csv(config.PREDICTIONS_DIR / "exp_76_diversity_blend_validation.csv",
              index=False)

    W.to_csv(config.ARTIFACTS_DIR / "exp76_cross_window.csv", index=False)
    S.to_csv(config.ARTIFACTS_DIR / "exp76_seeds.csv", index=False)
    R.update({"cross_window": rows, "wins": wins, "mean_dRMSE": mean_d,
              "mean_dMAE": mean_dm, "mean_resid_corr": mean_rho,
              "seeds": seed_rows, "seed_wins": seed_wins,
              "criteria": crit, "accepted": accepted,
              "w_frontiers": frontiers, "leakage_checks": leak,
              "primary_diagnostics": d})
    (config.ARTIFACTS_DIR / "exp76_summary.json").write_text(
        json.dumps(R, indent=2, default=str), encoding="utf-8")
    print(f"\n  total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
