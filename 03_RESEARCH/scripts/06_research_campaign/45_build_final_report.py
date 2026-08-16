"""
Build the FINAL MODEL PERFORMANCE REPORT (markdown + PDF) from the audited
comparison table produced by 44_final_performance_report.py.

REPORTING ONLY. Trains nothing. Reads only artifacts already on disk.

    python scripts/06_research_campaign/45_build_final_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config
from pipeline.report_pdf import render_markdown_to_pdf

ART = config.ARTIFACTS_DIR
OUT_MD = config.REPORTS_DIR / "FINAL_MODEL_PERFORMANCE_REPORT.md"
OUT_PDF = config.REPORTS_DIR / "FINAL_MODEL_PERFORMANCE_REPORT.pdf"
THRESHOLD = 0.5


def f4(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def cls(v):
    return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def main():
    T = pd.read_csv(ART / "final_performance_comparison.csv")
    r77 = json.loads((config.EXPERIMENTS_DIR /
                      "exp_77_recursive_member_upgrade.json").read_text(encoding="utf-8"))
    r76 = json.loads((config.EXPERIMENTS_DIR /
                      "exp_76_architectural_diversity_blend.json").read_text(encoding="utf-8"))
    r79 = json.loads((config.EXPERIMENTS_DIR /
                      "exp_79_upgrade_seed_check.json").read_text(encoding="utf-8"))
    team = json.loads((ART / "team_doc_analysis.json").read_text(encoding="utf-8"))
    op = pd.DataFrame(r77["operating_point"])
    op = op[op.pair == "AB2"]
    w77 = pd.DataFrame(json.loads(
        (ART / "exp77_summary.json").read_text(encoding="utf-8"))["windows"])

    L = []
    A = L.append

    A("# Final Model Performance Report")
    A("")
    A("**Project:** NPN_HACKATHON — Walmart M5 store-item demand forecasting  ")
    A("**Task:** frozen-origin 28-day-ahead forecasting  ")
    A("**Scope of this document:** reporting and verification only. No model was "
      "trained, changed or re-selected to produce it.")
    A("")

    # ------------------------------------------------------------------
    A("## 1. Validation setup")
    A("")
    A("| Item | Value |")
    A("|---|---|")
    A("| Forecast origin | `d_1913` (2016-04-24) |")
    A("| Predicted days | `d_1914` – `d_1941` (2016-04-25 → 2016-05-22) |")
    A("| Horizon | 28 days, generated in one shot from the origin |")
    A("| Series | 30,490 store-item combinations |")
    A("| Predictions scored | **853,720** (30,490 × 28) |")
    A("| Ground truth | real observed sales; held out, never used in training |")
    A("| Metric basis | **validation only** — no ground truth exists after `d_1941` |")
    A("")
    A("Every model in the comparison table below was scored on this identical "
      "set of 853,720 predictions, with no weighting and no series excluded. "
      "The protocol is strict: at origin *T* the model emits all 28 days, and "
      "nothing from *T+1…T+28* may enter any feature, model choice, calibration "
      "or blend weight.")
    A("")

    # ------------------------------------------------------------------
    A("## 2. Comparison table")
    A("")
    A("Regression metrics (RMSE, MAE) are the task metrics. Accuracy / Precision "
      "/ Recall / F1 are **classification-style demand-occurrence diagnostics**, "
      "defined in section 3 — they are not what any model was trained to "
      "optimise and must not be read as overall accuracy.")
    A("")
    A("| Model | Objective | RMSE | MAE | Accuracy (Demand > 0) | Precision | Recall | F1 Score |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in T.iterrows():
        name = r["Model"]
        if r["Role"] == "FINAL SHIPPED CHAMPION":
            name = f"**{name}** *(FINAL SHIPPED CHAMPION)*"
        elif r["Role"] == "ORIGINAL CHAMPION":
            name = f"{name} *(original champion)*"
        elif r["Role"] == "SHAPE CHAMPION":
            name = f"{name} *(shape champion)*"
        A(f"| {name} | {r['Objective']} | {f4(r['RMSE'])} | {f4(r['MAE'])} | "
          f"{cls(r['Accuracy'])} | {cls(r['Precision'])} | {cls(r['Recall'])} | "
          f"{cls(r['F1'])} |")
    A("")
    A("**FINAL SHIPPED CHAMPION:** `0.60 × Direct 38-feature model + 0.40 × "
      "Recursive-shape 32-feature model`, RMSE **2.0929**, MAE **1.0395**.")
    A("")
    A("### Why four rows show N/A")
    A("")
    A("N/A means the metric was **not evaluated**, never estimated. Occurrence "
      "metrics require per-row predictions, and four models do not have them "
      "on disk:")
    A("")
    A("- **Naive last value / rolling mean 7 / rolling mean 28** — scored in "
      "Stage 1 from the registry; prediction files were not retained.")
    A("- **Shape+Cycle 38 features** — see the data-integrity note in section 6.")
    A("- **Diversity blend w=0.60 (the shipped model)** — Experiment #77 "
      "persisted metrics and per-window summaries but not per-row predictions. "
      "Computing its occurrence metrics would require retraining both members, "
      "which is a modelling run and out of scope here. The nearest saved "
      "artifacts are the w=0.50 blend and the 36-feature shape model, both "
      "reported above.")
    A("")

    # ------------------------------------------------------------------
    A("## 3. The demand-occurrence rule")
    A("")
    A("A single rule, applied identically to every model with saved predictions:")
    A("")
    A("```")
    A("actual  event : y_true  > 0      (at least one unit actually sold)")
    A(f"predicted event : y_pred >= {THRESHOLD}    (point forecast rounds to >= 1 unit)")
    A("```")
    A("")
    A(f"The {THRESHOLD} cut is the only non-arbitrary threshold for a count "
      "target: it is the value at which a forecast rounds to one unit. It was "
      "fixed once and never tuned per model.")
    A("")
    A("| Quantity | Value |")
    A("|---|---|")
    A("| Rows with actual demand > 0 | 388,995 of 853,720 |")
    A("| Base rate | **45.56%** |")
    A("| Rows with actual demand = 0 | 464,725 (54.44%) |")
    A("")
    A("**These models were never trained to classify.** They minimise Tweedie "
      "deviance on a zero-inflated count target, so the occurrence metrics are "
      "a by-product of thresholding a regression output. A model could improve "
      "F1 while getting materially worse at the actual task.")
    A("")

    # ------------------------------------------------------------------
    A("## 4. Why RMSE and MAE are the primary metrics")
    A("")
    A("The task is **how many units will sell**, not *whether any will*. "
      "Inventory decisions need the quantity: ordering 3 when 11 sell is a "
      "stockout, and both a \"correct\" occurrence classification.")
    A("")
    A("- **RMSE** is the headline because it penalises large misses "
      "quadratically, and the business cost of demand error is convex — the "
      "expensive failures are the big ones. It is also the metric the whole "
      "campaign was pre-registered against.")
    A("- **MAE** is reported alongside because RMSE alone can be dominated by a "
      "small number of high-volume series. In this dataset the top volume "
      "decile carries **66%** of squared error, so RMSE and MAE can and do move "
      "in opposite directions. Reporting one without the other hides real "
      "trade-offs — including the one this project's final model makes.")
    A("")
    A("### Why there is no single valid \"accuracy %\"")
    A("")
    A("1. **The target is a count, not a class.** Accuracy needs a discretisation "
      "that the task does not supply; every threshold gives a different number.")
    A("2. **A trivial model scores well.** 54.44% of rows are genuine zeros, so "
      "\"always predict no demand\" scores 54.44% accuracy while being useless. "
      "Our models reach ~70% — the honest comparison is against 54.44%, not 0%.")
    A("3. **It discards magnitude entirely.** Predicting 1 when 40 sold counts "
      "as a correct positive.")
    A("4. **It is threshold-dependent and therefore gameable.** Lowering the cut "
      "raises recall and accuracy on this class balance without improving any "
      "forecast.")
    A("")
    A("The occurrence metrics are included because they were asked for and are "
      "genuinely informative about *one* aspect of behaviour — note the "
      "recursive member's distinctly high recall (0.8148) and low precision "
      "(0.6216) against the direct models, which is a real behavioural "
      "difference and part of why blending the two works. They are not a "
      "ranking metric.")
    A("")

    # ------------------------------------------------------------------
    A("## 5. Evidence behind the shipped champion")
    A("")
    A("The headline 2.0929 is one window. The model was accepted on four "
      "windows and multiple seeds, with criteria fixed before each run.")
    A("")
    A("### Per-window, at the shipped weight w = 0.60")
    A("")
    A("| Window | Dates | RMSE | MAE | ΔRMSE vs direct | ΔMAE vs direct |")
    A("|---|---|---|---|---|---|")
    dates = {"primary_spring_2016": "2016-04-25 → 05-22",
             "christmas_2015": "2015-12-12 → 2016-01-08",
             "summer_2015": "2015-07-16 → 08-12",
             "autumn_2015": "2015-10-02 → 10-29"}
    for _, r in op.iterrows():
        A(f"| {r['window']} | {dates[r['window']]} | {r['RMSE']:.4f} | "
          f"{r['MAE']:.4f} | {r['dRMSE_vs_A']:+.4f} | {r['dMAE_vs_A']:+.4f} |")
    A(f"| **Mean** | | | | **{op.dRMSE_vs_A.mean():+.4f}** | "
      f"**{op.dMAE_vs_A.mean():+.4f}** |")
    A("")
    A("### High-volume decile (66% of all squared error)")
    A("")
    A("| Window | Direct champion | Shipped blend | Δ |")
    A("|---|---|---|---|")
    for _, r in w77.iterrows():
        A(f"| {r['window']} | {r['A_highvol']:.4f} | {r['blend_AB2_highvol']:.4f} | "
          f"{r['blend_AB2_highvol'] - r['A_highvol']:+.4f} |")
    A("")
    A("Improves on **every** window — the error concentration no earlier "
      "experiment managed to move.")
    A("")
    A("### Acceptance record")
    A("")
    A("| Experiment | Test | Result |")
    A("|---|---|---|")
    A(f"| #76 | blend beats direct, 4 windows | {r76['window_wins']}/4 |")
    A(f"| #76 | 3 seeds, both members reseeded | {r76['seed_wins']}/3 |")
    A("| #76 | negative control (same architecture, reseeded) | "
      f"{r76['negative_control']['same_architecture_gain']:+.4f} vs "
      f"{r76['negative_control']['diversity_gain']:+.4f} — "
      f"{r76['negative_control']['gain_attributable_to_architecture']:+.4f} "
      "attributable to architecture |")
    A(f"| #77 | member upgrade, 4 windows | {r77['blend_wins']}/4, mean "
      f"{r77['mean_blend_dRMSE']:+.4f} |")
    A(f"| #79 | seed stability, 6 (window, seed) cells | "
      f"{r79['blend_wins']}/6 blend, {r79['member_wins']}/6 member |")
    A("")
    A("Leakage was verified structurally at every window and at the forecast "
      "origin: all 38 direct-model features are bit-identical when every day "
      "after the origin is overwritten, and the recursive member's working "
      "matrix provably never contains post-origin actuals.")
    A("")

    # ------------------------------------------------------------------
    A("## 6. Data-integrity finding")
    A("")
    A("The audit that produced this report re-derived every RMSE and MAE from "
      "the saved prediction files and compared them against the experiment "
      "registry. It failed on first run and surfaced a real defect:")
    A("")
    A("`predictions/validation/exp_74_new_champion_validation.csv` is **not** "
      "the 38-feature champion's output. It is a byte-identical copy of "
      "`exp_72_shape_validation.csv` (same MD5), i.e. the **36-feature** shape "
      "model. Script `36_exp74_reproduce_and_extend.py` discards the "
      "38-feature model's predictions at line 166 and writes Part A's at "
      "line 243.")
    A("")
    A("**What this does and does not affect:**")
    A("")
    A("- The champion's registry metrics (RMSE 2.1157 / MAE 1.0287) are "
      "**correct** — they were measured on the 38-feature model and have since "
      "been reproduced bit-identically (drift 0.00e+00).")
    A("- The mislabelled file was used as \"the champion\" in the headroom "
      "diagnostic that motivated Experiment #76. Its conclusion — that the "
      "recursive model is the standout blend partner — was then confirmed "
      "directly by retraining in #76 and #77, so nothing downstream rests on "
      "the mislabelled file.")
    A("- The 38-feature champion has **no** saved per-row predictions, which is "
      "why its occurrence metrics are N/A above.")
    A("")
    A("Closing this properly requires one reproduction run to regenerate and "
      "correctly name that prediction file. That is a modelling run and was not "
      "performed for this report.")
    A("")

    # ------------------------------------------------------------------
    A("## 7. Non-comparable reference — the other team's reported result")
    A("")
    A("**The other team's reported RMSE ≈ 2.0324 is not comparable to any "
      "number in this report and must not be placed in the same table.**")
    A("")
    A("Their pipeline recomputes rolling and lag features *inside* the forecast "
      "horizon using actual sales from the validation window. Measured "
      "directly against their feature definitions "
      "(`experiments/artifacts/team_doc_analysis.json`):")
    A("")
    A("| Their feature | Days of the 28-day horizon that use future actuals |")
    A("|---|---|")
    for k in ("rolling_mean_7", "rolling_mean_28", "rolling_zero_count_7",
              "lag_7", "lag_28"):
        A(f"| `{k}` | **{team['leak_days_out_of_28'][k]} / 28** |")
    A("")
    A("A model that may read the answers for 27 of the 28 days it is predicting "
      "is solving a materially easier problem — closer to one-day-ahead "
      "forecasting with a rolling update than to 28-day-ahead forecasting from "
      "a frozen origin. Our own leakage probe, run deliberately and marked "
      "`DO_NOT_USE`, scored **1.9165** by permitting a similar violation; that "
      "figure is recorded in the ledger precisely so the gap is understood as a "
      "methodology difference and not a modelling one.")
    A("")
    A("For a like-for-like view, `model_08_team_style_reproduction` implements "
      "their described approach under our frozen-origin rules and scores "
      "**2.1835** on the table above.")
    A("")

    # ------------------------------------------------------------------
    A("## 8. Known limitations of the shipped model")
    A("")
    A("1. **MAE regresses.** +0.0186 mean across four windows against the direct "
      "champion. Deliberate and disclosed: the blend is RMSE-optimal. The full "
      "weight frontier is in `exp77_operating_point.csv`; w = 0.65 trades "
      "−0.0234 RMSE for +0.0158 MAE if a different balance is wanted.")
    A("2. **The #77 gain is concentrated.** Autumn (−0.0105) and Christmas "
      "(−0.0053) carry it; the primary and summer windows contributed −0.0005 "
      "and −0.0004, which is noise. On summer the upgraded member was actually "
      "*worse* than the one it replaced.")
    A("3. **Cost.** Two models instead of one — roughly double the training and "
      "inference budget, and the recursive member takes ~7 minutes to build "
      "because it retrains on 420 daily origins.")
    A("4. **Seed evidence is partial.** #76's blend has a full 3-seed leg; "
      "#77's upgrade was seed-checked on the two windows that carry its effect "
      "(6/6 cells), not on all four.")
    A("5. **The champion's 2.1157 is its most favourable draw.** Observed "
      "single-seed range was 2.1157–2.1211, so paired comparisons within a run "
      "are the honest ones, and this report quotes them that way in section 5.")
    A("")

    # ------------------------------------------------------------------
    A("## 9. Sources")
    A("")
    A("Every figure traces to an artifact under version control:")
    A("")
    A("| Content | Path |")
    A("|---|---|")
    A("| Comparison table, as printed above | `experiments/artifacts/final_comparison_table.csv` |")
    A("| Same table plus audit columns and confusion counts | `experiments/artifacts/final_performance_comparison.csv` |")
    A("| Audit + table build | `scripts/06_research_campaign/44_final_performance_report.py` |")
    A("| Per-experiment records | `experiments/registry/*.json` (79 records) |")
    A("| Experiment index | `experiments/EXPERIMENT_LEDGER.md` |")
    A("| Blend acceptance | `exp_76_architectural_diversity_blend.json` |")
    A("| Member upgrade + weight frontier | `exp_77_recursive_member_upgrade.json`, `exp77_operating_point.csv` |")
    A("| Seed stability | `exp_79_upgrade_seed_check.json` |")
    A("| Leakage analysis of the team approach | `experiments/artifacts/team_doc_analysis.json` |")
    A("| Shipped forecast | `predictions/final_forecast/final_forecast_28day_v3_diversity_blend.csv` |")
    A("")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {OUT_MD}")

    render_markdown_to_pdf(
        OUT_MD, OUT_PDF,
        title="Final Model Performance Report",
        subtitles=["NPN_HACKATHON — Walmart M5 demand forecasting",
                   "Frozen-origin 28-day-ahead validation, 853,720 predictions",
                   "Shipped champion: 0.60 x Direct(38f) + 0.40 x Recursive-shape(32f)"],
        footer="NPN_HACKATHON — Final Model Performance Report")
    print(f"  wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
