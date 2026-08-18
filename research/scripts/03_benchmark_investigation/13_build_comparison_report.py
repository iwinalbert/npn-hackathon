
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

from pipeline import charts, config, experiment
from pipeline.report_pdf import render_markdown_to_pdf

EXPS = {r["experiment_name"]: r for r in experiment.load_all()}

TEAM = {
    "LightGBM (Tweedie)": (2.0324, 1.0869),
    "Random Forest (MSE)": (2.0770, 1.1187),
    "XGBoost (Poisson)": (2.1434, 1.1275),
}


def M(name, key):
    r = EXPS.get(name)
    return None if r is None else r.get("metrics", {}).get(key)


def scatter_chart() -> str:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    ours = [
        ("Our Model 4 (safe, verified)", M("model_04_tweedie_recency_listing", "RMSE"),
         M("model_04_tweedie_recency_listing", "MAE"), charts.GOOD, "o"),
        ("Our Model 2 (safe)", M("model_02_tweedie", "RMSE"),
         M("model_02_tweedie", "MAE"), charts.ACCENT, "o"),
        ("Team-style reproduction (safe)", M("model_08_team_style_reproduction", "RMSE"),
         M("model_08_team_style_reproduction", "MAE"), charts.ACCENT, "s"),
        ("LEAKY probe (unsafe, diagnostic)",
         M("diagnostic_leakage_probe_DO_NOT_USE", "RMSE"),
         M("diagnostic_leakage_probe_DO_NOT_USE", "MAE"), charts.BAD, "X"),
    ]
    for lab, r, m, c, mk in ours:
        if r is None:
            continue
        ax.scatter(r, m, s=110, color=c, marker=mk, zorder=3, edgecolor="white", lw=1.2)
        ax.annotate(lab, (r, m), textcoords="offset points", xytext=(9, 5),
                    fontsize=8, color="#222")

    for lab, (r, m) in TEAM.items():
        ax.scatter(r, m, s=110, facecolor="none", edgecolor="#c8860d",
                   marker="D", lw=1.8, zorder=3)
        ax.annotate(f"TEAM: {lab}", (r, m), textcoords="offset points",
                    xytext=(9, -12), fontsize=8, color="#8a5c00")

    ax.set_xlabel("RMSE  (lower is better)")
    ax.set_ylabel("MAE  (lower is better)")
    ax.set_title("Every measured result, plus the team's reported figures",
                 fontsize=10.5, color=charts.NAVY, loc="left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = charts.CHART_DIR / "team_comparison_scatter.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return "charts/team_comparison_scatter.png"


def main():
    ours_r = M("model_04_tweedie_recency_listing", "RMSE")
    ours_m = M("model_04_tweedie_recency_listing", "MAE")
    rep_r = M("model_08_team_style_reproduction", "RMSE")
    rep_m = M("model_08_team_style_reproduction", "MAE")
    leak_r = M("diagnostic_leakage_probe_DO_NOT_USE", "RMSE")
    leak_m = M("diagnostic_leakage_probe_DO_NOT_USE", "MAE")

    tier_path = config.ARTIFACTS_DIR / "team_style_by_volume_tier.csv"
    tier = pd.read_csv(tier_path) if tier_path.exists() else None
    pow_path = config.ARTIFACTS_DIR / "tweedie_power_probe.csv"
    powdf = pd.read_csv(pow_path) if pow_path.exists() else None

    chart = scatter_chart()

    L = ["# Team Benchmark — Fair Comparison Investigation", "",
         f"*Generated {date.today().isoformat()}. Every figure attributed to us "
         "comes from an experiment that actually ran and is recorded in "
         "`experiments/`. The three team figures are quoted exactly as supplied.*",
         ""]

    L += ["> **Headline: the comparison is NOT currently fair, and we could not "
          "reproduce the team's result.** Their methodology is not documented "
          "anywhere in this project or on this machine, and their reported "
          "RMSE/MAE combination was not reproduced by any of four methodologies "
          "we tested. This report explains exactly what we ruled out and what "
          "would settle the question.", ""]

    L += ["> **Terms.** **RMSE** punishes large misses much more heavily than "
          "small ones. **MAE** is the plain average miss. **Leakage** means "
          "letting information into the model that would not have existed when "
          "the forecast was really made — for example using \"yesterday's sales\" "
          "for day 20 of a 28-day forecast, when yesterday is itself still in the "
          "future. **Tweedie** is a loss function for non-negative data with many "
          "zeros.", "", "---", ""]

    L += ["## 1. What the team did — what we could actually find", "",
          "**We searched for it properly before assuming anything.** A full-text "
          "search across every file in the project for their reported numbers "
          "(2.0324, 1.0869, 2.0770, 1.1187, 2.1434, 1.1275) and for the strings "
          "`RandomForest`, `XGBoost`, `Random Forest` returned only:", "",
          "- coincidental digit sequences inside our own EDA tables and per-series "
          "statistics (e.g. a series whose standard deviation happens to be 2.0324)",
          "- our own model files, prediction files and reports",
          "- one docstring in `pipeline/metrics.py` that we wrote ourselves", "",
          "We also searched outside the project — Desktop, Downloads and the wider "
          "OneDrive folder — for notebooks, scripts, model files or result files. "
          "Nothing relating to the team's models exists on this machine.", "",
          "> **Therefore: the team's methodology is UNDOCUMENTED and could not be "
          "inspected.** Everything below labelled \"team\" is either a number they "
          "supplied or an explicitly-flagged reconstruction. We did not invent "
          "their method and we do not claim to know it.", "",
          "### What we know versus what we do not", "",
          "| Question | Status |", "|---|---|",
          "| Their reported RMSE / MAE | **Known** — supplied by the team |",
          "| That they used the same raw and processed datasets | **Stated** by the team |",
          "| Training cutoff / validation start / validation end | **UNKNOWN** |",
          "| Number of forecast days | **UNKNOWN** |",
          "| Number of series scored | **UNKNOWN** |",
          "| Direct or recursive forecasting | **UNKNOWN** |",
          "| Whether predictions were clipped at zero | **UNKNOWN** |",
          "| Whether zero rows were dropped or reweighted | **UNKNOWN** |",
          "| Exact lag / rolling feature definitions | **UNKNOWN** |",
          "| LightGBM parameters and Tweedie variance power | **UNKNOWN** |",
          "| How RMSE and MAE were computed (pooled? per-series? aggregated?) | **UNKNOWN** |",
          ""]

    L += ["## 2. Feature comparison", "",
          "Step 1 of the brief asked for a feature-by-feature table. We can fill "
          "in our column honestly and must leave theirs unknown — writing anything "
          "else would be inventing their method.", "",
          "| Feature | Used by team? | Used by us? | Difference | Reproduce? |",
          "|---|---|---|---|---|"]
    feat_rows = [
        ("lag_1", "us: yes (origin-relative)"), ("lag_7", "us: yes (origin-relative)"),
        ("lag_14", "us: yes (origin-relative)"), ("lag_21", "us: no"),
        ("lag_28", "us: yes (origin-relative)"), ("lag_35", "us: no (tested in reproduction)"),
        ("lag_56", "us: no (tested in reproduction)"),
        ("rolling_mean_7", "us: yes"), ("rolling_mean_14", "us: no"),
        ("rolling_mean_28", "us: yes — our single strongest feature (74% of model gain)"),
        ("rolling_mean_56", "us: no (tested in reproduction)"),
        ("rolling_std_7", "us: yes"), ("rolling_std_28", "us: yes"),
        ("rolling_min / rolling_max", "us: no (tested in reproduction)"),
        ("price features", "us: yes — sell_price, recent avg, relative price, missing flag"),
        ("calendar features", "us: yes — weekday, month, year, weekend"),
        ("SNAP", "us: yes — matched to each series' own state"),
        ("events", "us: yes — event_name_1/2, event_type_1/2"),
        ("store / category / dept / item / state ids", "us: yes — all five, native categoricals"),
        ("target encoding", "us: no"),
        ("recency (days_since_last_sale, zero_streak)", "us: yes — measured as no help"),
        ("listing-aware (pre_listing, days_since_first_listing)", "us: yes — measured as no help"),
    ]
    for f, note in feat_rows:
        us = note.split("us: ")[1]
        L.append(f"| `{f}` | **UNKNOWN** | {us} | cannot be determined | "
                 "not possible without their spec |")
    L += ["", "> Every cell in the \"Used by team?\" column is unknown for the same "
          "reason: there is no artefact of their work to read. This table is "
          "included because the brief asked for it, not because it tells us "
          "anything about them.", ""]

    L += ["## 3. Is the validation identical?", "",
          "**Unknown, and this matters more than anything else in this report.** "
          "Our setup is fully specified and was held constant across every "
          "experiment we have ever run:", "",
          "| | Ours |", "|---|---|",
          "| Forecast origin | d_1913 (2016-04-24) |",
          "| Validation days | d_1914 .. d_1941 (2016-04-25 .. 2016-05-22) |",
          "| Horizon | 28 days |", "| Series | 30,490 |",
          "| Predictions scored | 853,720 |",
          "| Forecasting mode | direct multi-horizon, fixed origin |",
          "| Clipping | predictions clipped at 0 |",
          "| Metric | pooled over all rows, unweighted |", "",
          "We did not change any of this, per the brief. If the team used a "
          "different window, a different horizon, a subset of series, or a "
          "different metric implementation, then the two sets of numbers are not "
          "measuring the same thing and no arithmetic comparison between them is "
          "meaningful.", ""]

    L += ["## 4. What we actually ran", "",
          "Four configurations, all scored on our validation window, all on the "
          "same 853,720 predictions, all with the same metric code.", "",
          "| Configuration | RMSE | MAE | Leakage-safe? |", "|---|---|---|---|",
          f"| Diagnostic leaky probe (deliberately unsafe) | {leak_r:.4f} | {leak_m:.4f} | **NO — by design** |",
          "| *Team reported (their setup)* | *2.0324* | *1.0869* | *unknown* |",
          f"| **Our best model (Model 4)** | **{ours_r:.4f}** | **{ours_m:.4f}** | **yes — verified** |",
          f"| Team-style reproduction (28-day lookback) | {rep_r:.4f} | {rep_m:.4f} | yes — verified |",
          "", f"![Comparison]({chart})", "",
          "The team's other two models sit at RMSE 2.0770 / MAE 1.1187 (Random "
          "Forest, MSE) and RMSE 2.1434 / MAE 1.1275 (XGBoost, Poisson).", ""]

    L += ["## 5. Diagnosing the difference — what we ruled out", "",
          "Rather than guess, we tested the plausible explanations one at a time.",
          "",
          "### Ruled out: prediction calibration or clipping", "",
          "We rescaled our existing predictions by every constant factor from 0.9 "
          "to 2.0. The best RMSE any rescaling can achieve is **2.1195** — still "
          "well above their 2.0324. Scaling up made *both* metrics worse, not one "
          "better. So their result is not our model with a different multiplier or "
          "clipping rule; the prediction vector itself must be different.", "",
          "### Ruled out: a different validation window", "",
          "Their MAE (1.0869) is *higher* than ours (1.0319). MAE tracks the "
          "demand level of the window being scored. We measured the mean daily "
          "sales of every 28-day window in the last two years: the range is 1.0622 "
          "to **1.4428**, and the maximum is our own window. For their MAE to come "
          "from a higher-demand window at our error rate, that window would need a "
          "mean of about 1.52 — higher than any window that exists in the data. So "
          "window choice alone cannot produce their numbers.", "",
          "### Ruled out: per-target-day lag construction", "",
          "The most common public M5 recipe builds one row per (series, day) with "
          "lags of 28 days or more, rather than computing features once at the "
          "origin. We implemented it (`pipeline/team_style.py`, 25 features, "
          "21,312,510 training rows), verified it leakage-safe with the same "
          "corruption test, and held the objective, hyperparameters and validation "
          f"window identical to ours. It scored **{rep_r:.4f} / {rep_m:.4f}** — "
          "worse than our model on both metrics.", ""]

    if tier is not None:
        L += ["Broken down by how much each series normally sells:", "",
              "| Volume tier | Rows | Actual mean | Our RMSE | Team-style RMSE | Difference |",
              "|---|---|---|---|---|---|"]
        for _, r in tier.iterrows():
            L.append(f"| {r['tier']} | {int(r['n']):,} | {r['actual_mean']:.3f} | "
                     f"{r['ours_RMSE']:.4f} | {r['teamstyle_RMSE']:.4f} | "
                     f"{r['dRMSE']:+.4f} |")
        L += ["", "The 28-day-lookback version is worse everywhere and worst on "
              "high-volume series. That is the explanation: our origin-relative "
              "features include `lag_1` and `rolling_mean_7` measured right up to "
              "the forecast origin, which is fresher information than a 28-day-old "
              "lag. Freshness matters most for the busiest products.", ""]

    L += ["### Not ruled out: a per-target-day leak", "",
          "The remaining common explanation for a score that cannot be reproduced "
          "legitimately is leakage — computing `lag_1` or `rolling_mean_7` "
          "relative to each *target* day instead of the forecast origin. On day 20 "
          "of the horizon that reads a real sales value from inside the validation "
          "window, which nobody would have had on the day the forecast was made.",
          "",
          "We built exactly that, confirmed it leaky with the corruption test "
          "(10 features moved when the future was altered), and measured it: "
          f"**RMSE {leak_r:.4f}, MAE {leak_m:.4f}**.", "",
          "> **How to read that, carefully.** A leak of this kind scores *better* "
          f"than the team's reported RMSE ({leak_r:.4f} vs 2.0324), so it is "
          "*sufficient* to produce a number in their range. That is **not** "
          "evidence that they leaked. It only establishes that their RMSE is "
          "reachable by a mechanism we know produces invalid results, and is not "
          "reachable by any valid mechanism we tested. It is a reason to check, "
          "not an accusation.", "",
          "### The part that no explanation covers", "",
          "Their MAE (1.0869) is worse than **every** configuration we measured — "
          f"worse than our best ({ours_m:.4f}), worse than the safe team-style "
          f"reproduction ({rep_m:.4f}), and worse than the leaky probe "
          f"({leak_m:.4f}). Meanwhile their RMSE is better than both of our "
          "legitimate models. Lower RMSE with higher MAE means comparatively fewer "
          "large misses but more medium-sized ones, and we could not produce that "
          "combination with any of four methodologies. This is the strongest "
          "single sign that their numbers were produced under a different "
          "evaluation setup rather than simply by a better model.", ""]

    L += ["## 6. Comparison table, with the honest labels", "",
          "| Approach | RMSE | MAE | Features | Validation window | Notes |",
          "|---|---|---|---|---|---|",
          "| Team reported — LightGBM Tweedie | 2.0324 | 1.0869 | unknown | **unknown** | reported by team; not independently verified |",
          "| Team reported — Random Forest MSE | 2.0770 | 1.1187 | unknown | **unknown** | reported by team |",
          "| Team reported — XGBoost Poisson | 2.1434 | 1.1275 | unknown | **unknown** | reported by team |",
          f"| Our current best — LightGBM Tweedie | {ours_r:.4f} | {ours_m:.4f} | 32 | d_1914..d_1941, 30,490 series | leakage-verified |",
          f"| Our team-style reproduction | {rep_r:.4f} | {rep_m:.4f} | 25 | d_1914..d_1941, 30,490 series | leakage-verified reconstruction |",
          f"| Diagnostic leaky probe | {leak_r:.4f} | {leak_m:.4f} | 25 | d_1914..d_1941, 30,490 series | **invalid — diagnosis only** |",
          "",
          "### Why we are not reporting a percentage difference", "",
          "The brief asks for a percentage improvement or degradation, but only "
          "after establishing that the methodology matches. It does not match — "
          "or rather, we cannot establish that it matches, which for this purpose "
          "is the same thing. Computing "
          f"`(2.0324 - {ours_r:.4f}) / 2.0324 = {(2.0324 - ours_r) / 2.0324 * 100:+.2f}%` "
          "would imply the two numbers measure the same quantity on the same rows. "
          "They may not. The arithmetic is shown here so nobody has to wonder what "
          "it would have been, and it should not be quoted as a result.", ""]

    if powdf is not None:
        cur = powdf[powdf.power == 1.1].iloc[0]
        best = powdf.sort_values("inner_RMSE").iloc[0]
        L += ["## 7. The one improvement lever we tested", "",
              "Our error analysis established that high-volume series are 7.7% of "
              "rows but carry **61% of all squared error**, and that we "
              "systematically under-predict them. The Tweedie variance power "
              "controls how much the objective concentrates on zeros versus the "
              "tail, and we had never tested it — 1.1 was an untested assumption "
              "sitting directly on the thing limiting our RMSE.", "",
              "Tested on the **inner** window (d_1886..d_1913) so the primary "
              "window stays an unbiased estimate:", "",
              "| Tweedie power | Inner RMSE | Inner MAE | Bias | High-volume RMSE | High-volume bias |",
              "|---|---|---|---|---|---|"]
        for _, r in powdf.sort_values("power").iterrows():
            mark = " *(current)*" if r["power"] == 1.1 else ""
            L.append(f"| {r['power']}{mark} | {r['inner_RMSE']:.4f} | "
                     f"{r['inner_MAE']:.4f} | {r['bias']:+.4f} | "
                     f"{r['high_vol_RMSE']:.4f} | {r['high_vol_bias']:+.4f} |")
        if abs(best["power"] - 1.1) < 1e-9:
            L += ["", "**Result: our existing setting of 1.1 was the best of those "
                  "tested.** No change is recommended. This is a negative result "
                  "and it is reported as one.", ""]
        else:
            L += ["", f"On the inner window, power {best['power']} looked like a "
                  f"clear win: RMSE improved by "
                  f"{best['inner_RMSE'] - cur['inner_RMSE']:+.4f}, MAE improved, "
                  "and — exactly as the error analysis predicted — the "
                  "high-volume bias shrank from +0.317 to +0.129. Every signal "
                  "pointed the same way.", ""]

        applied = config.ARTIFACTS_DIR / "tweedie_power_applied.json"
        if applied.exists():
            ap = json.loads(applied.read_text(encoding="utf-8"))
            a = ap["model_9_power_1_5"]
            hv = ap["high_volume"]
            L += ["### Then we tested it on the primary window — and it did not hold",
                  "",
                  "Because the power was selected using only the inner window, "
                  "applying it once to d_1914..d_1941 is a clean unbiased test. "
                  "We ran it:", "",
                  "| | RMSE | MAE | High-volume RMSE | High-volume bias |",
                  "|---|---|---|---|---|",
                  "| Model 4 — power 1.1 | **2.1210** | 1.0319 | 5.9756 | −0.389 |",
                  f"| Model 9 — power 1.5 | {a['RMSE']:.4f} | **{a['MAE']:.4f}** | "
                  f"{hv['RMSE']:.4f} | {hv['bias']:+.3f} |",
                  f"| Change | {a['RMSE'] - 2.1210:+.4f} | {a['MAE'] - 1.0319:+.4f} | "
                  f"{hv['RMSE'] - 5.9756:+.4f} | — |", "",
                  "**The improvement did not transfer.** RMSE got slightly worse, "
                  "and the high-volume bias moved the wrong way on this window "
                  "(−0.42) even though it had improved on the other one (+0.13). "
                  "A −0.0133 gain on one 28-day window turned into a +0.0053 loss "
                  "on the next.", "",
                  "> **Decision: do not change the Tweedie power.** Model 4 with "
                  "power 1.1 remains our best model. This is the correct outcome "
                  "of a disciplined process — we formed a hypothesis from measured "
                  "evidence, tested it properly, and it failed. Had we selected on "
                  "the primary window instead, we would have shipped a change that "
                  "was really just noise.", "",
                  "It is also a caution about the team comparison itself: a 0.013 "
                  "swing between adjacent 28-day windows is ordinary noise here, "
                  "and the gap being discussed is only about 0.09.", ""]

    L += ["## 8. What we should change, and what we should not", "",
          "### Should NOT change", "",
          "- **Our validation setup.** It is fully specified, leakage-verified, "
          "and consistent across every experiment. Changing it to match an unknown "
          "setup would destroy the one thing we can actually defend.",
          "- **Our origin-relative feature design.** We tested the main "
          f"alternative and it was worse ({rep_r:.4f} vs {ours_r:.4f}).",
          "- **Do not adopt per-target-day lags to chase the benchmark.** They are "
          "only better when they reach into the forecast window, which is exactly "
          "the thing that makes a forecast worthless in production.",
          "- **Do not re-add recency or listing features as novelty.** Previously "
          "measured as no help; nothing here changes that.", "",
          "### Should change / do next, in priority order", "",
          "1. **Ask the team for five specific things** — their validation dates, "
          "series count, horizon, whether lags are computed relative to the target "
          "day or the origin, and their metric code. Four of the five are one-line "
          "answers, and they would settle this entirely. The harness to run a real "
          "head-to-head already exists.",
          "2. **Attack the high-volume tail.** It is 7.7% of rows and 61% of our "
          "error. Options with evidence behind them: volume-weighted training, a "
          "separate model for the high tier, or per-horizon models.",
          "3. **Test recursive forecasting.** The leaky probe is not just a "
          f"diagnostic — it is an upper bound. It says that perfect knowledge of "
          f"recent sales during the horizon would be worth about "
          f"{ours_r - leak_r:.4f} RMSE. A recursive strategy feeds the model's own "
          "predictions back in as lags, which is legitimate and captures some "
          "fraction of that headroom. That is the single most promising legitimate "
          "direction this investigation has produced.", ""]

    L += ["## 9. Is the comparison genuinely fair?", "",
          "**No, and it cannot be made fair from our side alone.** We reproduced "
          "everything reproducible: same dataset, same window, same metric code, "
          "same 853,720 predictions, and a good-faith reconstruction of the "
          "standard public recipe. What we cannot reproduce is a methodology we "
          "have never seen.", "",
          "What we can state with confidence:", "",
          "- Our result is leakage-verified by an empirical corruption test. Their "
          "leakage status is unknown.",
          "- Our number is reproducible: an independently retrained run matched it "
          "to four decimal places.",
          "- Under our methodology, our model has the better MAE and their reported "
          "RMSE is lower.",
          "- The RMSE/MAE combination they report was not reproducible by any of "
          "four methodologies we tested, which suggests a difference in evaluation "
          "rather than purely in modelling.", "",
          "> **We are not claiming to beat the team, and we are not conceding that "
          "they beat us.** Neither claim is supportable on the evidence. What is "
          "supportable is that our pipeline is verified, reproducible and honest "
          "about its own limits — and that a genuine comparison is five questions "
          "away.", ""]

    L += ["---", "",
          "*All figures attributed to us come from executed runs recorded in "
          "`experiments/`: `model_04_tweedie_recency_listing`, "
          "`model_08_team_style_reproduction`, "
          "`diagnostic_leakage_probe_DO_NOT_USE`, and `probe_tweedie_power_*`. "
          "No existing model, experiment or report was modified to produce this "
          "comparison.*"]

    md = config.REPORTS_DIR / "TEAM_FAIR_COMPARISON_REPORT.md"
    pdf = config.REPORTS_DIR / "TEAM_FAIR_COMPARISON_REPORT.pdf"
    md.write_text("\n".join(L), encoding="utf-8")
    render_markdown_to_pdf(
        md, pdf, title="Team Benchmark — Fair Comparison Investigation",
        subtitles=["M5 Retail Demand Forecasting — Problem Statement 11",
                   "Can we reproduce the team's result? What explains the difference?",
                   "NPN AIA Hackathon — St. Joseph's College of Engineering"],
        footer="TEAM_FAIR_COMPARISON_REPORT.pdf — methodology not fully reproduced")
    print(f"wrote {md.name} and {pdf.name}")


if __name__ == "__main__":
    main()
