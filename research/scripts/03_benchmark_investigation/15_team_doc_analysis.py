
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

from pipeline import charts, config, metrics, models
from pipeline.data_loader import M5Data
from pipeline.features import FEATURE_GROUPS
from pipeline.report_pdf import render_markdown_to_pdf

DOC = config.PROJECT_ROOT / "end_to_end_approach.md"
OURS_PRED = config.PREDICTIONS_DIR / "model_04_tweedie_recency_listing_validation.csv"

OUR_RMSE, OUR_MAE = 2.1210, 1.0319
TEAM_RMSE, TEAM_MAE = 2.0324, 1.0869
LEAKY_RMSE, LEAKY_MAE = 1.9165, 0.9754
TEAMSTYLE_RMSE, TEAMSTYLE_MAE = 2.1835, 1.0498


def run_diagnostics() -> dict:
    d = M5Data(load_prices=False)
    S = d.sales_wide
    P = pd.read_csv(OURS_PRED)
    y = P.y_true.to_numpy(float)
    p = P.y_pred.to_numpy(float)
    si = P.series_idx.to_numpy()
    ti = P.target_day_idx.to_numpy()

    roll28 = np.empty(len(y))
    prev3 = np.empty(len(y), bool)
    for k, (s, t) in enumerate(zip(si, ti)):
        roll28[k] = S[s, t - 28:t].mean()
        prev3[k] = (S[s, t - 3:t] == 0).all()
    ghost = (roll28 > 3) & (y == 0) & prev3

    tot_sq = ((y - p) ** 2).sum()
    cat = d.series_meta.cat_id.to_numpy()[si]
    series_tot = pd.Series(y).groupby(si).transform("sum").to_numpy()

    variants = {
        "all rows (our reported figure)": np.ones(len(y), bool),
        "exclude ghost-stockout rows": ~ghost,
        "exclude every zero-actual row": y > 0,
        "exclude series with no sales in the window": series_tot > 0,
        "FOODS only": cat == "FOODS",
    }
    vres = {}
    for lab, m in variants.items():
        vres[lab] = {"n": int(m.sum()),
                     "RMSE": metrics.rmse(y[m], p[m]),
                     "MAE": metrics.mae(y[m], p[m])}

    leak_map = {
        "lag_7": [h > 7 for h in range(1, 29)],
        "lag_28": [h > 28 for h in range(1, 29)],
        "rolling_mean_7": [h > 1 for h in range(1, 29)],
        "rolling_mean_28": [h > 1 for h in range(1, 29)],
        "rolling_zero_count_7": [h > 1 for h in range(1, 29)],
    }
    leak_summary = {k: int(sum(v)) for k, v in leak_map.items()}

    return {
        "ghost_stockout": {
            "rows_flagged": int(ghost.sum()),
            "pct_of_rows": round(float(ghost.mean() * 100), 4),
            "mean_prediction_on_those_rows": round(float(p[ghost].mean()), 3),
            "share_of_total_squared_error_pct": round(
                float(((y[ghost] - p[ghost]) ** 2).sum() / tot_sq * 100), 3),
        },
        "evaluation_population_variants": vres,
        "leak_days_out_of_28": leak_summary,
        "leak_map": {k: [bool(x) for x in v] for k, v in leak_map.items()},
    }


def chart_leak_map(leak_map: dict) -> str:
    feats = list(leak_map.keys())
    fig, ax = plt.subplots(figsize=(9.0, 2.6))
    grid = np.array([leak_map[f] for f in feats], dtype=float)
    ax.imshow(grid, aspect="auto", cmap=matplotlib.colors.ListedColormap(
        ["#2e7d5b", "#a8443c"]), vmin=0, vmax=1)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8.5)
    ax.set_xticks([0, 6, 13, 20, 27])
    ax.set_xticklabels(["day 1", "day 7", "day 14", "day 21", "day 28"], fontsize=8)
    ax.set_xlabel("Forecast horizon day")
    ax.set_title("If features are rebuilt for each target day: which days would read "
                 "future sales\n(red = reads a day inside the forecast window)",
                 fontsize=9.5, color=charts.NAVY, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    p = charts.CHART_DIR / "leak_window_map.png"
    fig.savefig(p, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return "charts/leak_window_map.png"


def chart_candidates() -> str:
    labels = ["Leaky probe\n(measured, unsafe)", "TEAM reported",
              "Ours — Model 4\n(measured, safe)", "Team-style 28d lag\n(measured, safe)"]
    vals = [LEAKY_RMSE, TEAM_RMSE, OUR_RMSE, TEAMSTYLE_RMSE]
    cols = [charts.BAD, "#c8860d", charts.GOOD, charts.ACCENT]
    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    b = ax.bar(range(4), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_ylim(1.8, 2.28)
    ax.set_title("Where the team's reported RMSE sits among our measured results",
                 fontsize=10, color=charts.NAVY, loc="left")
    fig.tight_layout()
    p = charts.CHART_DIR / "team_doc_candidates.png"
    fig.savefig(p, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return "charts/team_doc_candidates.png"


def build_report(diag: dict, c1: str, c2: str) -> None:
    g = diag["ghost_stockout"]
    v = diag["evaluation_population_variants"]
    lk = diag["leak_days_out_of_28"]

    L: list[str] = []
    A = L.append

    A("# The Team's Approach vs Our Pipeline")
    A("")
    A(f"*Generated {date.today().isoformat()}. Read-only analysis: no model was "
      "trained, no pipeline code changed, no existing report overwritten.*")
    A("")
    A("> **The single most important thing to understand about the reference "
      "document.** `end_to_end_approach.md` is a **plan and a pitch**, not a "
      "record of what was built. It contains no train/test split, no validation "
      "dates, no forecast horizon, no hyperparameters, no Tweedie power, and no "
      "RMSE or MAE figures anywhere. The only metric it names is **WRMSSE**, not "
      "RMSE/MAE. So the numbers 2.0324 and 1.0869 **cannot be traced to this "
      "document at all** — it does not tell us how they were produced.")
    A("")
    A("> **Plain-English glossary.** **RMSE** = average error, with big misses "
      "punished much more heavily. **MAE** = plain average error. **Leakage** = "
      "the model accidentally sees information from the future that would not "
      "have existed when the forecast was really made. **Lag** = a past value "
      "(lag_7 = sales seven days earlier). **Rolling mean** = average over a "
      "recent stretch of days. **Tweedie** = a loss function suited to data that "
      "is never negative and is mostly zeros. **SNAP** = a US food-assistance "
      "benefit; the calendar records which days it was usable in each state.")
    A("")
    A("---")
    A("")

    A("## Part 1 — What the team's document actually says")
    A("")
    A("| Feature / Method | Purpose | Team approach | Evidence in document | We use it? | Worth testing? |")
    A("|---|---|---|---|---|---|")
    rows = [
        ("lag_7", "weekly momentum", "sales 7 days ago", "Phase 1 feature table",
         "Yes — but origin-relative", "Already have"),
        ("lag_28", "monthly momentum", "sales 28 days ago", "Phase 1 feature table",
         "Yes", "Already have"),
        ("rolling_mean_7", "recent trend", "7-day average", "Phase 1 feature table",
         "Yes — window ends at origin", "Already have"),
        ("rolling_mean_28", "recent trend", "28-day average", "Phase 1 feature table",
         "Yes — our strongest feature (74% of gain)", "Already have"),
        ("rolling_zero_count_7", "intermittency", "count of zero days in last 7",
         "Phase 1 feature table", "**No**", "**Yes — cheap, plausible**"),
        ("day_of_week", "weekly rhythm", "weekday index", "Phase 1 feature table",
         "Yes (`wday`)", "Already have"),
        ("month", "seasonality", "calendar month", "Phase 1 feature table",
         "Yes", "Already have"),
        ("day_of_month", "payday effect", "day number within month",
         "Phase 1 feature table", "**No**", "**Yes — cheap, genuinely missing**"),
        ("is_weekend", "weekend surge", "Sat/Sun flag; cites 31% surge",
         "Phase 1 feature table", "Yes", "Already have"),
        ("SNAP", "benefit-day demand", "`snap_CA/TX/WI` as three columns",
         "Phase 1 feature table", "Yes — but matched to each series' own state",
         "Ours is stricter; no change"),
        ("SNAP x FOOD interaction", "food-specific SNAP lift",
         "`is_food_and_is_snap`; cites 10.2%", "Phase 1 feature table",
         "**Not explicit** (model can learn it from snap + cat_id)",
         "**Yes — cheap**"),
        ("price_pct_change", "price momentum", "week-over-week % change",
         "Phase 1 feature table", "**No** (we use price ÷ own recent average)",
         "**Yes — different construction**"),
        ("phantom promotion", "proxy for missing promo data",
         "flag weeks where price fell >5%", "Phase 1 feature table + Idea #3",
         "**No**", "Maybe — see Part 4"),
        ("ghost stockout", "exclude suspicious zeros",
         "28-day avg >3/day AND today 0 AND prior 3 days 0; **exclude from training**",
         "Phase 0 Step 2 + Phase 1", "**No**", "Test as a *feature*, not a deletion"),
        ("leading-zero removal", "drop pre-launch rows",
         "mark days before first price as `pre_launch`; **remove entirely**",
         "Phase 0 Step 1", "We flag it, never delete", "Test removal from training only"),
        ("Christmas override", "domain rule",
         "force prediction to 0 on every Dec 25", "Phase 0 Step 3",
         "**No**", "**No — see Part 4**"),
        ("cat_id / dept_id / store_id", "hierarchy", "categoricals as embeddings",
         "Phase 1 + Phase 2", "Yes — plus `item_id` and `state_id`", "Already broader"),
        ("Global LightGBM", "one model for all series", "single global model",
         "Phase 2", "Yes", "Already have"),
        ("Tweedie", "zero-inflated loss", "Tweedie objective; power not stated",
         "Phase 2", "Yes (power 1.1; 1.3/1.5 also tested)", "Already tested"),
        ("Foods-first tuning", "tune on the volume driver",
         "tune hyperparameters on FOODS; others on defaults", "Phase 2",
         "**No**", "Low priority — see Part 4"),
        ("Bottom-up reconciliation", "coherent hierarchy",
         "sum store-item preds up 12 levels", "Phase 2", "**No**",
         "**No — not required by the deliverable**"),
    ]
    for r in rows:
        A("| " + " | ".join(r) + " |")
    A("")
    A("### Claims in the document that do not match the data")
    A("")
    A("These were checked directly against the raw files. They matter because "
      "three of them are the exact figures that were in dispute at the start of "
      "this project — this document is where they came from.")
    A("")
    A("| Document says | Actual, verified from raw files | Impact |")
    A("|---|---|---|")
    A("| \"FOODS drives **69.56%** of total sales volume\" | **68.62%** | "
      "Cosmetic. Does not change any modelling decision. |")
    A("| Melting creates \"**~30 million** rows\" | **59,181,090** rows | "
      "Cosmetic, but roughly 2x out — worth correcting in the pitch. |")
    A("| \"training **42,840** separate models (one per series)\" | **30,490** "
      "series. 42,840 is the sum of all 12 WRMSSE hierarchy levels | "
      "Cosmetic, but a judge who knows M5 may notice. |")
    A("| \"The sales file has `d_1` to **`d_1969`** as column headers\" | Sales "
      "files stop at **`d_1941`**; only `calendar.csv` reaches `d_1969` | "
      "**Not cosmetic.** Melting to d_1969 would create 28 days of phantom rows "
      "with no sales. If those were filled with 0 and trained on, it would harm "
      "the model; if used as the prediction frame, it is harmless. |")
    A("| \"**10.2%** SNAP shockwave\" | Our EDA measured **+12.7%** overall and "
      "**+17.3%** within FOODS | The 10.2% figure closely matches an early "
      "CA-only spot check in `DATASET_SUMMARY.md`, so it looks like a "
      "state-specific number quoted as a global one. |")
    A("")

    A("## Part 2 — What our pipeline actually does")
    A("")
    A("Every value below is read from the repository, not from memory.")
    A("")
    A("| Setting | Value | Source |")
    A("|---|---|---|")
    A(f"| Forecast origin | d_{config.VALIDATION_ORIGIN_IDX + 1} (2016-04-24) | `pipeline/config.py` |")
    A("| Validation days | d_1914 .. d_1941 (2016-04-25 .. 2016-05-22) | `pipeline/config.py` |")
    A(f"| Horizon | {config.HORIZON} days | `pipeline/config.py` |")
    A(f"| Series | {config.N_SERIES:,} | verified against raw files |")
    A(f"| Predictions scored | {config.N_SERIES * config.HORIZON:,} | all series x all 28 days |")
    A("| Training window | 15 origins, d_1493 .. d_1885 (420 contiguous days) | `experiments/model_04_*.json` |")
    A("| Training rows | 12,805,800 | same |")
    A(f"| Lags | {config.LAGS} — **origin-relative** | `pipeline/config.py` |")
    A(f"| Rolling windows | {config.ROLLING_WINDOWS} (mean and std), ending at the origin | `pipeline/config.py` |")
    A("| Recency | days_since_last_sale, zero_streak_length, days_since_first_sale | `pipeline/features.py` |")
    A("| Listing | days_since_first_listing, pre_listing | `pipeline/features.py` |")
    A("| Price | sell_price, recent_avg_price, price ÷ recent avg, price_is_missing | `pipeline/features.py` |")
    A("| Hierarchy | item_id, dept_id, cat_id, store_id, state_id (native categoricals) | `pipeline/features.py` |")
    A("| Objective | tweedie, variance_power = 1.1 | `experiments/model_04_*.json` |")
    A(f"| Rounds / leaves / lr | {models.N_ESTIMATORS} / {models.DEFAULT_PARAMS['num_leaves']} / {models.DEFAULT_PARAMS['learning_rate']} | `pipeline/models.py` |")
    A(f"| Seed | {config.RANDOM_SEED}, `deterministic=True` | `pipeline/models.py` |")
    A("| Clipping | predictions clipped at 0 | `pipeline/models.py` |")
    A("| Strategy | direct multi-horizon, fixed origin (no recursion) | `pipeline/backtest.py` |")
    A("| Leakage test | future sales overwritten with 9999; all 32 features bit-identical | `pipeline/validation_checks.py` |")
    A("")
    A("**Total features: 32, in 7 groups.**")
    A("")
    for gname, cols in FEATURE_GROUPS.items():
        A(f"- **{gname}** — {', '.join('`' + c + '`' for c in cols)}")
    A("")

    A("## Part 3 — Side by side")
    A("")
    A("| # | Topic | Team document | Our pipeline | Difference | Should test? |")
    A("|---|---|---|---|---|---|")
    sbs = [
        ("1", "Feature definitions", "listed by name only, no formulas",
         "explicit formulas, unit-checked", "theirs under-specified", "n/a"),
        ("2", "Lag definitions", "lag_7, lag_28 in a melted long table",
         "lag_1/7/14/28 **frozen at the origin**",
         "**the deepest difference — see Part 5**", "**already measured**"),
        ("3", "Rolling windows", "rolling_mean_7/28 in a melted table",
         "windows **end at the origin**, held constant", "same as above", "measured"),
        ("4", "Origin-relative?", "**not mentioned anywhere**", "yes, enforced and tested",
         "unknown vs verified", "n/a"),
        ("5", "Price features", "price_pct_change (week over week)",
         "price ÷ own 8-week average, plus missing flag",
         "different construction", "**Yes — cheap**"),
        ("6", "SNAP", "three raw columns snap_CA/TX/WI",
         "one flag matched to each series' own state",
         "ours is stricter", "No — ours is better"),
        ("7", "Calendar", "day_of_week, month, day_of_month, is_weekend",
         "wday, month, year, is_weekend, 4 event fields",
         "**they have day_of_month, we do not**", "**Yes — cheap**"),
        ("8", "Intermittency", "rolling_zero_count_7",
         "days_since_last_sale, zero_streak_length",
         "different encoding of the same idea", "**Yes — cheap**"),
        ("9", "Ghost stockouts", "**delete those rows from training**",
         "never delete anything", "philosophical", "Test as a feature only"),
        ("10", "Leading zeros", "**delete pre-launch rows**", "flag, never delete",
         "philosophical", "Test removal from training"),
        ("11", "Christmas", "hard override to 0 on Dec 25", "no override",
         "**irrelevant here — neither window contains Dec 25**", "**No**"),
        ("12", "Categoricals", "cat_id, dept_id, store_id",
         "those plus item_id and state_id", "ours is broader", "No"),
        ("13", "Objective", "Tweedie, power unstated",
         "Tweedie, power 1.1 (1.3 and 1.5 also measured)", "we tested more", "Done"),
        ("14", "Hyperparameters", "**not stated**", "fully recorded", "unknown vs known", "n/a"),
        ("15", "Training sample", "everything except deleted rows",
         "15 origins x 28 days = 420 contiguous days", "different sampling",
         "Maybe — more history"),
        ("16", "Validation", "**not stated anywhere**",
         "fixed origin d_1913, 28 days, 30,490 series", "**unknown vs fully specified**",
         "**must ask them**"),
        ("17", "Forecast strategy", "not stated", "direct multi-horizon", "unknown", "n/a"),
        ("18", "Clipping", "not stated", "clip at 0", "unknown", "n/a"),
        ("19", "Reconciliation", "bottom-up across 12 levels", "none",
         "not required by submission format", "**No**"),
        ("20", "Foods-first tuning", "tune on FOODS, defaults elsewhere",
         "one global setting", "different tuning target", "Low priority"),
    ]
    for r in sbs:
        A("| " + " | ".join(r) + " |")
    A("")

    A("## Part 4 — Which of their ideas actually hold up")
    A("")
    A("Ratings: **A** = strongly supported, safe to test · **B** = reasonable, "
      "needs testing · **C** = risky/unverified · **D** = do not use without "
      "much better evidence.")
    A("")
    A("### Global LightGBM — **A**")
    A("Correct, and we already do it. One model across 30,490 series lets sparse "
      "items borrow patterns from thousands of others. Their reasoning is sound "
      "(though the count is 30,490, not 42,840).")
    A("")
    A("### Tweedie — **A**")
    A("Correct and independently confirmed by us: switching only the objective "
      "improved our RMSE from 2.1467 to 2.1256. Their sentence \"no other loss "
      "function handles this correctly\" is too strong — Poisson and plain "
      "regression both work, just less well — but the choice is right.")
    A("")
    A("### SNAP x FOOD interaction — **B**")
    A("The effect is real: our EDA measured +12.7% overall and +17.3% within "
      "FOODS, and it lands exactly where domain knowledge predicts. A tree model "
      "can already discover this from `snap` and `cat_id` together, so an "
      "explicit product term may add little — but it is one line of code and "
      "worth a test.")
    A("")
    A("### Leading-zero (pre-launch) removal — **B for training, D for evaluation**")
    A("The underlying fact is real and we confirmed it *more strongly* than they "
      "did: rows before an item's first recorded price have a **100.00%** "
      "zero-sales rate. Removing them from **training** is defensible. But two "
      "things must be said plainly. First, at our forecast origin **0% of rows "
      "are pre-launch**, so this cannot change the forecast — by 2016 every item "
      "has long since launched. Second, removing them from **evaluation** would "
      "silently change the denominator and make scores incomparable.")
    A("")
    A("### Ghost stockout detection — **C**")
    A("The rule (28-day average >3/day, today 0, previous 3 days 0) is a "
      "reasonable heuristic, but the document calls the result a stockout. **The "
      "dataset has no inventory field, so a stockout cannot be confirmed.** A "
      "genuinely dead item, a delisting, or a bad demand week all produce the "
      "same pattern.")
    A("")
    A(f"We applied their exact rule to our validation window: it flags "
      f"**{g['rows_flagged']:,} rows ({g['pct_of_rows']}%)**, on which our model "
      f"predicts {g['mean_prediction_on_those_rows']} units against an actual of "
      f"zero. Those rows carry **{g['share_of_total_squared_error_pct']}%** of "
      "our total squared error.")
    A("")
    A("> **This matters for the comparison.** Deleting rows from *training* is a "
      "modelling choice. Deleting them from *evaluation* is not — it removes "
      "precisely the rows a model is worst on. We measured that: excluding them "
      f"moves our RMSE from **{OUR_RMSE:.4f}** to "
      f"**{v['exclude ghost-stockout rows']['RMSE']:.4f}**, about "
      f"{(OUR_RMSE - v['exclude ghost-stockout rows']['RMSE']) / (OUR_RMSE - TEAM_RMSE) * 100:.0f}% "
      "of the gap to their figure. Real, but not the main story — and it moves "
      "MAE *down* to "
      f"{v['exclude ghost-stockout rows']['MAE']:.4f}, away from their higher MAE.")
    A("")
    A("### Phantom promotion — **C**")
    A("A price drop >5% is a *price drop*, not a confirmed promotion. Our own EDA "
      "found sales rose after both price increases (+71%) and decreases (+48%) "
      "with a median effect of zero, meaning price changes often coincide with "
      "something else rather than causing it. Usable as a weak signal; must never "
      "be described to judges as promotion detection.")
    A("")
    A("### Christmas override — **D (for this task)**")
    A("The finding is real — Christmas is −99.95% versus a local baseline, stores "
      "are shut. But **neither our validation window (2016-04-25 to 2016-05-22) "
      "nor the actual forecast window (2016-05-23 to 2016-06-19) contains a "
      "December 25.** The override cannot change a single prediction. It is a "
      "good slide and a zero-impact feature; spending time on it would be time "
      "not spent on the 61% of error sitting in high-volume series.")
    A("")
    A("### Foods-first tuning — **C**")
    A("Justified in the document by WRMSSE being volume-weighted. But we are "
      "being compared on plain RMSE and MAE, which are not volume-weighted, so "
      "the premise does not transfer. FOODS does dominate our error (74% of it), "
      "so weighting *might* help — but that is a different argument than the one "
      "the document makes.")
    A("")
    A("### Bottom-up hierarchical reconciliation — **D**")
    A("`sample_submission.csv` asks only for store-item forecasts. Summing them "
      "upward is arithmetic that changes **none** of the 853,720 numbers being "
      "scored. It cannot improve RMSE or MAE by even a rounding error. It is "
      "presentation value only.")
    A("")

    A("## Part 5 — What could explain their 2.0324")
    A("")
    A("Ranked by how much evidence supports each, highest first. **Nothing here "
      "is proven.** The document is silent on validation, so every explanation is "
      "a hypothesis about a method we have never seen.")
    A("")
    A(f"![Where their number sits]({c2})")
    A("")
    A("### 1. Features rebuilt per target day, without freezing at the origin — **most likely**")
    A("")
    A("This is the one hypothesis the document actively supports. It says: melt "
      "to long format, one row per (item, store, day), then build `lag_7`, "
      "`rolling_mean_7`, `rolling_mean_28`, `rolling_zero_count_7`. It never "
      "mentions freezing those values at a forecast origin.")
    A("")
    A("If features are computed per row in a melted table and the model then "
      "predicts 28 days at once, most of the horizon reads sales that had not "
      "happened yet:")
    A("")
    A("| Feature | Reads | Leaks on |")
    A("|---|---|---|")
    A(f"| `lag_7` | day t−7 | days {lk['lag_7']} of 28 (from day 8 onward) |")
    A(f"| `rolling_mean_7` | days t−7 … t−1 | days {lk['rolling_mean_7']} of 28 (from day 2 onward) |")
    A(f"| `rolling_mean_28` | days t−28 … t−1 | days {lk['rolling_mean_28']} of 28 |")
    A(f"| `rolling_zero_count_7` | days t−7 … t−1 | days {lk['rolling_zero_count_7']} of 28 |")
    A(f"| `lag_28` | day t−28 | **0 of 28 — safe** |")
    A("")
    A(f"![Leak window map]({c1})")
    A("")
    A("We already measured what this is worth. Our deliberately-leaky diagnostic "
      f"probe scored **RMSE {LEAKY_RMSE:.4f}** where our safe model scores "
      f"**{OUR_RMSE:.4f}**. Their **{TEAM_RMSE}** sits between the two — and that "
      "is exactly where a *milder* leak would land, because their shortest lag is "
      "`lag_7` rather than the `lag_1` our probe used.")
    A("")
    A("> **Stated carefully: this is a hypothesis, not an accusation.** The "
      "document does not say how they validated. It is entirely possible they "
      "froze features correctly and simply did not write it down. What we can say "
      "is that the mechanism is consistent with everything the document does "
      "describe, and it is the only explanation we have that produces a number in "
      "their range. It is a reason to ask them one specific question, not a "
      "verdict.")
    A("")
    A("### 2. A different evaluation population — **plausible, partly measured**")
    A("")
    A("The document instructs that pre-launch rows and ghost-stockout rows be "
      "removed. If that removal also touched the evaluation set, the two scores "
      "are not measuring the same rows. We measured several variants on our own "
      "predictions:")
    A("")
    A("| Rows scored | n | RMSE | MAE |")
    A("|---|---|---|---|")
    for lab, r in v.items():
        A(f"| {lab} | {r['n']:,} | {r['RMSE']:.4f} | {r['MAE']:.4f} |")
    A("")
    A("Ghost-stockout exclusion moves RMSE in the right direction but explains "
      "only a fraction of the gap, and pushes MAE the wrong way.")
    A("")
    A("### 3. A different validation window — **largely ruled out**")
    A("")
    A("Their MAE is *higher* than ours, and MAE scales with how busy the period "
      "is. We measured the mean daily sales of every 28-day window in the last "
      "two years: the range is 1.0622 to **1.4428**, and the highest is our own "
      "window. There is no busier window for them to have used.")
    A("")
    A("### 4. Genuinely different features — **possible but small**")
    A("")
    A("They have three features we lack: `day_of_month`, `rolling_zero_count_7`, "
      "and an explicit SNAP×FOODS term. Our own ablation showed that everything "
      "beyond recent-demand features moved RMSE by hundredths, so a realistic "
      "expectation here is 0.00–0.02, not 0.09.")
    A("")
    A("### 5. Hyperparameters — **possible, unknown**")
    A("Theirs are not stated. Ours are untuned by design. Our capacity search "
      "found more rounds made things *worse*, so this is unlikely to be worth 0.09.")
    A("")
    A("### 6. A different metric implementation — **cannot be excluded**")
    A("The document names **WRMSSE** as the metric, not RMSE/MAE. If the reported "
      "figures were computed per-series and averaged, or on aggregated totals, or "
      "on a subset, they are simply a different quantity from ours.")
    A("")
    A("### Ruled out by measurement")
    A("- **Prediction clipping or calibration.** No rescaling of our predictions "
      f"reaches below {2.1195:.4f}; scaling up worsens both metrics.")
    A("- **Per-target-day features done *safely*** (28-day minimum lookback). We "
      f"built and measured it: **{TEAMSTYLE_RMSE:.4f}**, worse than ours.")
    A("- **Bottom-up reconciliation.** Mathematically cannot alter store-item "
      "predictions.")
    A("- **Christmas override.** Neither window contains December 25.")
    A("")
    A("> **The part still unexplained by any hypothesis.** Their MAE (1.0869) is "
      f"worse than every configuration we have measured — worse than ours "
      f"({OUR_MAE:.4f}), worse than the safe team-style build ({TEAMSTYLE_MAE:.4f}), "
      f"and worse than the leaky probe ({LEAKY_MAE:.4f}). Leakage improves both "
      "metrics, so leakage alone does not explain a *worse* MAE. The most likely "
      "reading is that two things differ at once: something that lowers their "
      "RMSE, and a base model or evaluation population that raises their MAE.")
    A("")

    A("## Part 6 — The experiment ladder, and what to skip")
    A("")
    A("All of these would use our exact validation window, unchanged target, and "
      "unchanged leakage controls, changing one factor at a time.")
    A("")
    A("| Exp | Change | Cost | Expected value | Verdict |")
    A("|---|---|---|---|---|")
    A("| A | Current best (reference) | none | — | **Already done** — RMSE 2.1210 |")
    A("| B | Ask the team 5 questions about their validation | minutes | **decisive** | **DO THIS FIRST** |")
    A("| C | + `day_of_month`, `rolling_zero_count_7`, SNAP×FOODS | ~5 min | small but real | **RUN** |")
    A("| D | + `price_pct_change` / phantom-promo flag | ~5 min | small | **RUN** (bundle with C) |")
    A("| E | Exclude pre-launch rows from **training** | ~5 min | small | **RUN** — settles a live question |")
    A("| F | Ghost-stockout flag as a **feature** (never deleted) | ~5 min | small | **RUN** (bundle with E) |")
    A("| G | Recursive forecasting (feed own predictions back as lags) | ~1–2 h | **largest legitimate upside** | **RUN if time** |")
    A("| H | Christmas override | ~5 min | **exactly zero** | **SKIP** — no Dec 25 in either window |")
    A("| I | Bottom-up reconciliation | ~30 min | **exactly zero on the metric** | **SKIP** for accuracy |")
    A("| J | Foods-first tuning | ~1 h | unclear | **SKIP for now** |")
    A("| K | Team categorical set (drop item_id/state_id) | ~5 min | likely negative | **SKIP** — ours is a superset |")
    A("")
    A("**Bundle C+D and E+F into two runs rather than four.** Our ablation showed "
      "individual feature groups move RMSE by hundredths, so testing them one at "
      "a time costs more time than the information is worth.")
    A("")

    A("## Part 7 — Recommended path")
    A("")
    A("### Keep from our pipeline")
    A("- The **fixed-origin design**. We tested the alternative and it was worse "
      f"({TEAMSTYLE_RMSE:.4f} vs {OUR_RMSE:.4f}).")
    A("- The **empirical leakage test**. It is the single most defensible thing "
      "in this project, and it already caught one real issue.")
    A("- **Tweedie**, global LightGBM, our broader categorical set, our "
      "state-matched SNAP flag, and clipping at zero.")
    A("- **Never deleting rows** from evaluation.")
    A("")
    A("### Borrow from the team")
    A("- `day_of_month` — genuinely missing from ours, and payday cycles are real.")
    A("- `rolling_zero_count_7` — a different, possibly better encoding of intermittency.")
    A("- Explicit SNAP×FOODS term — cheap to add.")
    A("- `price_pct_change` — a different price construction from ours.")
    A("- Pre-launch removal **from training only** — worth one clean test.")
    A("")
    A("### Test first")
    A("1. **Ask them the five questions** (Part 6, Exp B). Nothing we build "
      "competes with simply learning their validation setup.")
    A("2. **Bundle C+D** — four cheap features in one run.")
    A("3. **Bundle E+F** — training-set filtering.")
    A("4. **Recursive forecasting** if time remains.")
    A("")
    A("### Absolutely avoid")
    A("- **Copying their per-target-day lag construction to chase the score.** If "
      "our leading hypothesis is right, that number is only reachable by reading "
      "the future. A forecast that needs tomorrow's sales to predict tomorrow is "
      "worthless in production, and a judge who asks one careful question will "
      "expose it.")
    A("- **Deleting rows from the evaluation set.**")
    A("- **Christmas override and reconciliation** as accuracy work — both are "
      "provably zero-impact here.")
    A("- **Claiming ghost stockouts are detected.** No inventory field exists.")
    A("")
    A("### Highest-probability path to a genuinely better RMSE")
    A("Not feature dumping. Our error analysis is unambiguous: **high-volume "
      "series are 7.7% of rows and carry 61% of all squared error**, and we "
      "under-predict them (bias −0.389). The ranked options are:")
    A("")
    A("1. **Recursive forecasting** — the leaky probe puts an upper bound of "
      f"about {OUR_RMSE - LEAKY_RMSE:.2f} RMSE on the value of fresher in-horizon "
      "information. Recursion captures part of that legitimately.")
    A("2. **Volume-aware training** — weighting or a dedicated high-volume model.")
    A("3. **The cheap features above** — worth having, but expect hundredths.")
    A("")
    A("### Most likely to waste time")
    A("Christmas override, hierarchical reconciliation, phantom-promotion "
      "engineering, Foods-first tuning, and re-testing recency or listing "
      "features that we have already measured twice as no help.")
    A("")

    A("## The four questions, answered directly")
    A("")
    A("**What did the team do?** We know what they *planned*: melt to long "
      "format, build lag/rolling/calendar/price/SNAP features, delete pre-launch "
      "and suspected-stockout rows, train one global LightGBM with Tweedie loss, "
      "tune toward FOODS, reconcile bottom-up, and wrap it in an API, dashboard "
      "and GenAI copilot. We do **not** know how they validated it, and the "
      "document contains no RMSE or MAE at all.")
    A("")
    A("**What are we doing differently?** Chiefly one thing: we freeze every "
      "history-derived feature at the forecast origin and prove by experiment "
      "that no feature moves when the future is altered. We also never delete "
      "rows, we match SNAP to each series' own state, and we carry two extra "
      "categoricals.")
    A("")
    A("**What should we test next?** Ask them five questions; then one run adding "
      "`day_of_month` + `rolling_zero_count_7` + SNAP×FOODS + `price_pct_change`; "
      "then one run on training-set filtering; then recursive forecasting.")
    A("")
    A("**What could realistically close the 0.0886 RMSE gap?** On the evidence, "
      "possibly nothing — because the gap may not be real. Our best current "
      "explanation is that their features read into the forecast window, in which "
      "case the number is not reproducible by any valid method. Of the legitimate "
      "levers, recursion is the only one plausibly worth that much; the new "
      "features are worth hundredths, not tenths.")
    A("")
    A("**What should we not copy?** Per-target-day lags without origin freezing, "
      "row deletion from evaluation, the Christmas override, bottom-up "
      "reconciliation, and any language claiming stockouts or promotions have "
      "been detected.")
    A("")
    A("---")
    A("")
    A("*Read-only analysis. No model trained, no pipeline modified, no existing "
      "report overwritten. Measured figures come from `experiments/` and from "
      "recomputation over `predictions/model_04_tweedie_recency_listing_validation.csv`.*")

    md = config.REPORTS_DIR / "TEAM_APPROACH_VS_OUR_PIPELINE_REPORT.md"
    pdf = config.REPORTS_DIR / "TEAM_APPROACH_VS_OUR_PIPELINE_REPORT.pdf"
    md.write_text("\n".join(L), encoding="utf-8")
    render_markdown_to_pdf(
        md, pdf, title="The Team's Approach vs Our Pipeline",
        subtitles=["M5 Retail Demand Forecasting — Problem Statement 11",
                   "What they planned, what we built, and what could explain the difference",
                   "NPN AIA Hackathon — St. Joseph's College of Engineering"],
        footer="TEAM_APPROACH_VS_OUR_PIPELINE_REPORT.pdf — read-only analysis, no model trained")
    print(f"  wrote {md.name}")
    print(f"  wrote {pdf.name}")


def main():
    if not DOC.exists():
        raise SystemExit(f"reference document not found at {DOC}")
    print("Analysing team document against our pipeline (read-only)...")
    diag = run_diagnostics()
    (config.ARTIFACTS_DIR / "team_doc_analysis.json").write_text(
        json.dumps(diag, indent=2), encoding="utf-8")
    c1 = chart_leak_map(diag["leak_map"])
    c2 = chart_candidates()
    build_report(diag, c1, c2)
    print("  wrote artifacts/team_doc_analysis.json")


if __name__ == "__main__":
    main()
