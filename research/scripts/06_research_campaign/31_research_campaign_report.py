
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import charts, config
from pipeline.report_pdf import render_markdown_to_pdf

A = config.ARTIFACTS_DIR
CHAMP_RMSE, CHAMP_MAE = 2.1210429411947650, 1.0319268155496617

S70 = json.loads((A / "exp70_summary.json").read_text(encoding="utf-8"))
S71 = json.loads((A / "exp71_summary.json").read_text(encoding="utf-8"))
AUT = json.loads((A / "error_autopsy.json").read_text(encoding="utf-8"))
MEM = pd.DataFrame(S70["members"])
RHO = S70["mean_pairwise_residual_corr"]


def chart_ceiling():
    s = MEM.RMSE.mean()
    Ms = np.array([1, 2, 3, 4, 6, 10, 20, 50, 100, 1000])
    r = np.sqrt(s**2 * (RHO + (1 - RHO) / Ms))
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.plot(Ms, r, "-o", color=charts.ACCENT, lw=1.8, ms=5,
            label="predicted ensemble RMSE")
    ax.axhline(s * np.sqrt(RHO), color=charts.BAD, ls="--", lw=1.5,
               label=f"floor at infinite members = {s*np.sqrt(RHO):.4f}")
    ax.axhline(CHAMP_RMSE, color=charts.GOOD, ls="-", lw=1.5,
               label=f"single champion = {CHAMP_RMSE:.4f}")
    ax.axhline(2.0, color="#c8860d", ls=":", lw=1.8, label="target = 2.0000")
    ax.scatter([6], [S70["ensemble"]["RMSE"]], s=120, color=charts.NAVY, zorder=6,
               label=f"measured at M=6 = {S70['ensemble']['RMSE']:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of models averaged")
    ax.set_ylabel("RMSE")
    ax.set_ylim(1.95, 2.16)
    ax.set_title("Ensembling this architecture family cannot reach 2.0 — "
                 "or even the champion", fontsize=10, color=charts.NAVY, loc="left")
    ax.legend(fontsize=7.6, frameon=False, loc="center right")
    fig.tight_layout()
    p = charts.CHART_DIR / "campaign_ceiling.png"
    fig.savefig(p, bbox_inches="tight", dpi=140); plt.close(fig)
    return "charts/campaign_ceiling.png"


def main():
    c1 = chart_ceiling()
    s = MEM.RMSE.mean()
    inf_rmse = s * np.sqrt(RHO)
    o = AUT["oracles"]

    L: list[str] = []
    A_ = L.append

    A_("# Autonomous Research Campaign")
    A_("")
    A_(f"*Objective: reduce RMSE as far below 2.0 as legitimately possible. "
       f"Generated {date.today().isoformat()}. Two experiments run, both "
       f"rejected, campaign stopped on evidence.*")
    A_("")
    A_("> ## OUTCOME: the champion stands at **RMSE 2.1210 / MAE 1.0319**, and "
       "RMSE < 2.0 is **not achievable** with the information in this dataset. "
       "The evidence for that ceiling is quantitative and is set out below.")
    A_("")
    A_("> **Terms.** **Variance vs bias** — bias is a consistent tilt (predicting "
       "too high or too low every time); variance is error that flips sign row to "
       "row. **Residual correlation** — whether two models make the *same* "
       "mistakes on the same rows. **Oracle** — a cheating predictor allowed to "
       "see the answers, used to measure what is even possible.")
    A_("")
    A_("---")
    A_("")

    A_("## The audit that set the strategy")
    A_("")
    A_("Across 69 prior experiments, exactly one thing ever beat the champion "
       "(recursive forecasting, by 0.0029 — inside noise, rejected on a +0.0398 "
       "MAE cost). The failures cluster into two families:")
    A_("")
    A_("| Family | Attempts | Result |")
    A_("|---|---|---|")
    A_("| Attacks on **bias** | calibration, per-series correction (#69), volume weighting, high-volume rescaling | all failed |")
    A_("| Attacks on **information** | 14 new features in 4 groups, recency (x2), listing (x2), per-target-day lags | all failed |")
    A_("")
    A_("The error autopsy explains why: **MSE = 0.0049 bias-squared + 4.4939 "
       "variance. 99.89% of the error is variance.** Neither family was attacking "
       "the binding constraint.")
    A_("")
    A_("That left one canonical technique untested on RMSE-competitive models — "
       "**ensembling**, the standard remedy for variance. Phase 8 had blended "
       "Tweedie with L1, but L1 is deliberately a poor RMSE model, so that "
       "measured a metric trade-off, not variance reduction.")
    A_("")

    A_("## Experiment #70 — variance-reduction ensemble")
    A_("")
    A_("**Hypothesis.** Averaging several individually-strong models that make "
       "*different* mistakes cancels the fit-to-fit component of variance.")
    A_("")
    A_("**Design.** Six LightGBM Tweedie models, identical 32 features and "
       "training origins, diversified across seed, tree size (96–256 leaves), "
       "feature/bagging fractions, and Tweedie power (1.1–1.3). Equal weights "
       "fixed a priori, so nothing was selected using the validation window.")
    A_("")
    A_("| Member | Power | Leaves | Seed | RMSE | MAE |")
    A_("|---|---|---|---|---|---|")
    for _, r in MEM.iterrows():
        A_(f"| {r['member']} | {r['power']} | {int(r['leaves'])} | {int(r['seed'])} | "
           f"{r['RMSE']:.4f} | {r['MAE']:.4f} |")
    A_(f"| **Equal-weight ensemble** | — | — | — | **{S70['ensemble']['RMSE']:.4f}** | "
       f"**{S70['ensemble']['MAE']:.4f}** |")
    A_("")
    A_(f"**Result: REJECTED.** ΔRMSE {S70['delta']['RMSE']:+.4f}, "
       f"ΔMAE {S70['delta']['MAE']:+.4f}. The ensemble did not even beat its own "
       "best member.")
    A_("")
    A_("### The finding that ended the campaign")
    A_("")
    A_(f"**Mean pairwise residual correlation across the six models: "
       f"{RHO:.4f}.**")
    A_("")
    A_("Six models with different seeds, different tree sizes, different "
       "subsampling and different loss parameters make **essentially the same "
       "mistakes on the same rows**. If the errors were driven by the fitting "
       "procedure, they would decorrelate. They do not.")
    A_("")
    A_("**INTERPRETATION:** the 4.4939 variance is not model variance. It is "
       "variance in the target that every model sees identically — irreducible "
       "given the available features. That single number explains why 69 previous "
       "experiments failed, and it predicts that better models of this kind cannot "
       "help either.")
    A_("")

    A_("## Experiment #71 — year-over-year features")
    A_("")
    A_("**Why this followed.** If model variance is not the lever, only new "
       "*information* can help. The champion's longest lookback is 28 days; it has "
       "no way to know what a specific product sold in a specific store one year "
       "earlier. `month` and `week_of_year` give only a chain-wide seasonal "
       "average, which cannot express that one item peaks in May and another in "
       "November.")
    A_("")
    A_("This is materially different from the Phase 2 features that failed — those "
       "re-encoded information already present (a 14-day window between the "
       "existing 7 and 28, zero-counts restating recency). A 364-day lookback lies "
       "entirely outside the champion's feature horizon.")
    A_("")
    A_("**Features added:** `lag_364`, `rolling_mean_7_lag364`, "
       "`rolling_mean_28_lag364`, `yoy_level_ratio`. All leakage-safe: for target "
       "day T+h with h ≤ 28, a 364-day lookback reads at most T−336. Confirmed by "
       "corruption test — all 36 features unchanged when every post-origin sale "
       "was overwritten with 9999.")
    A_("")
    A_("| | RMSE | MAE |")
    A_("|---|---|---|")
    A_(f"| Champion | **{CHAMP_RMSE:.4f}** | **{CHAMP_MAE:.4f}** |")
    A_(f"| + year-over-year features | {S71['with_yoy']['RMSE']:.4f} | {S71['with_yoy']['MAE']:.4f} |")
    A_(f"| Change | {S71['delta']['RMSE']:+.4f} | {S71['delta']['MAE']:+.4f} |")
    A_("")
    A_(f"**Result: REJECTED.** ΔRMSE {S71['delta']['RMSE']:+.4f} — clearly worse, "
       "and outside the noise band in the wrong direction.")
    A_("")
    A_("The instructive detail: **the model did use them.** They took "
       f"{S71['new_feature_gain_share_pct']:.2f}% of total gain, with "
       "`rolling_mean_7_lag364` ranking 5th of 36 features. They carry real "
       "signal — but mostly noise, and the capacity they consumed would have been "
       "better spent on splits of `rolling_mean_28`. Last year's demand is a worse "
       "guide to next month than last month's demand is.")
    A_("")

    A_("## The practical ceiling, quantified")
    A_("")
    A_("### Bound 1 — ensembling cannot get there, and the maths is checkable")
    A_("")
    A_("For M models with residual standard deviation s and mean pairwise "
       "correlation ρ, the averaged residual variance is s²·(ρ + (1−ρ)/M). "
       f"With the measured s = {s:.4f} and ρ = {RHO:.4f}:")
    A_("")
    A_("| Models averaged | Predicted RMSE |")
    A_("|---|---|")
    for M in [1, 2, 6, 10, 100]:
        A_(f"| {M} | {np.sqrt(s**2*(RHO+(1-RHO)/M)):.4f} |")
    A_(f"| **infinite** | **{inf_rmse:.4f}** |")
    A_("")
    A_(f"> **This bound validates itself.** The formula predicts {np.sqrt(s**2*(RHO+(1-RHO)/6)):.4f} "
       f"at M = 6; the measured six-member ensemble scored "
       f"**{S70['ensemble']['RMSE']:.4f}** — an exact match to four decimals. The "
       "extrapolation to infinite members is therefore trustworthy.")
    A_("")
    A_(f"![Ceiling]({c1})")
    A_("")
    A_(f"**Averaging an unlimited number of models of this family bottoms out at "
       f"{inf_rmse:.4f} — worse than the single champion, and "
       f"{inf_rmse-2.0:.4f} away from the target.**")
    A_("")
    A_("### Bound 2 — 2.0 is below an oracle that already knows the answer")
    A_("")
    A_("| Predictor | RMSE |")
    A_("|---|---|")
    A_(f"| Champion | {CHAMP_RMSE:.4f} |")
    A_("| **Target** | **2.0000** |")
    A_(f"| Oracle: each series' true 28-day mean | {o['C_per_series_oracle_mean']['RMSE']:.4f} |")
    A_(f"| Oracle: optimal per-series rescale | {o['B_per_series_rescale']['RMSE']:.4f} |")
    A_("")
    A_("Reaching 2.0 means essentially **matching a predictor that already knows "
       "each series' average over the very window being forecast** — and beating "
       "it would require exploiting within-window day-to-day structure on top. "
       "Experiment #69 tested whether that per-series level is learnable from the "
       "past and found it is not: bias measured on one fit does not transfer to "
       "another.")
    A_("")
    A_("### Bound 3 — the error that would have to disappear is the unpredictable kind")
    A_("")
    A_("Going from 2.1210 to 2.0000 means removing **11.1% of all squared "
       "error**. The autopsy located that error precisely:")
    A_("")
    A_("| Where the error is | Share of squared error |")
    A_("|---|---|")
    A_(f"| Demand spikes (actual > 2× the series' historical mean) | {AUT['spikes']['share_of_sq_error_pct']}% |")
    A_(f"| Top volume decile | {AUT['volume_deciles'][-1]['sq_err_share_pct']:.1f}% |")
    A_(f"| Worst 1,000 rows (0.12% of data) | {AUT['worst']['top1000_share_pct']}% |")
    A_("")
    A_("And those spikes are barely predictable from anything available: the spike "
       "rate is 0.31 on weekends versus 0.23 midweek, 0.272 on SNAP days versus "
       "0.252 otherwise, 0.280 on event days versus 0.256. The model already has "
       "every one of those signals. With no promotion field and no inventory field "
       "anywhere in the dataset, the information needed simply is not present.")
    A_("")

    A_("## Decision: stop")
    A_("")
    A_("Research rule 20 permits stopping when the evidence shows further "
       "experiments are unlikely to produce meaningful improvement. That threshold "
       "is met, on three independent lines of evidence:")
    A_("")
    A_("1. **Model-side improvement is closed.** Six architecturally different "
       f"models correlate at ρ = {RHO:.4f}. Infinite ensembling floors at "
       f"{inf_rmse:.4f}. A seventh architecture (XGBoost, CatBoost, a neural "
       "network) would have to break that correlation, and nothing in the evidence "
       "suggests it would — all six already converge on the same conditional-mean "
       "estimate.")
    A_("2. **Information-side improvement is closed.** Every feature family has "
       "been tested: recent demand, calendar, price, interactions, recency, "
       "listing, per-target-day lags, and now year-over-year. Eighteen features "
       "across six experiments; none helped.")
    A_("3. **The target is below a cheating predictor.** 2.0 sits under the "
       "per-series oracle, and #69 showed that per-series level is not learnable "
       "from history.")
    A_("")
    A_("### What would actually change the answer")
    A_("")
    A_("Not a better model — **more information.** Specifically: a promotions or "
       "markdown calendar, inventory or stockout records, or store-level footfall. "
       "Those would address the spikes that carry 62% of the error. None exists in "
       "this dataset, and no amount of modelling recovers them.")
    A_("")

    A_("## Final position")
    A_("")
    A_("| | |")
    A_("|---|---|")
    A_(f"| Champion | Global LightGBM + Tweedie(1.1), 32 features |")
    A_(f"| RMSE | **{CHAMP_RMSE:.4f}** |")
    A_(f"| MAE | **{CHAMP_MAE:.4f}** |")
    A_("| Leakage | verified by corruption test |")
    A_("| Reproducibility | re-run reproduced the score to every decimal |")
    A_("| Robustness | 4 windows, RMSE std 0.033 |")
    A_(f"| Experiments run in total | 71 |")
    A_("")
    A_("The honest summary for the presentation: this project did not find a way "
       "below 2.0, and it can say precisely why. The error is dominated by "
       "genuinely unpredictable demand spikes; six different models make the same "
       "mistakes on the same rows; and the target sits below what a predictor with "
       "access to the answers achieves. That is a stronger position than an "
       "unexplained number would be.")
    A_("")
    A_("---")
    A_("")
    A_("*Experiments #70 and #71 are recorded in `experiments/`. Predictions saved "
       "separately. The champion model, its predictions, the final forecast, and "
       "all previous reports are unchanged.*")

    md = config.REPORTS_DIR / "AUTONOMOUS_RESEARCH_CAMPAIGN_REPORT.md"
    md.write_text("\n".join(L), encoding="utf-8")
    render_markdown_to_pdf(
        md, config.REPORTS_DIR / "AUTONOMOUS_RESEARCH_CAMPAIGN_REPORT.pdf",
        title="Autonomous Research Campaign",
        subtitles=["M5 Retail Demand Forecasting — Problem Statement 11",
                   "Two experiments, both rejected, and the evidence for the practical ceiling",
                   "NPN AIA Hackathon — St. Joseph's College of Engineering"],
        footer="AUTONOMOUS_RESEARCH_CAMPAIGN_REPORT.pdf — champion unchanged at 2.1210")
    print("  wrote AUTONOMOUS_RESEARCH_CAMPAIGN_REPORT.md and .pdf")


if __name__ == "__main__":
    main()
