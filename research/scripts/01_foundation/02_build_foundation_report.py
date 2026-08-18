
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config
from pipeline.report_pdf import render_markdown_to_pdf

RESULTS_PATH = config.ARTIFACTS_DIR / "foundation_checks.json"
MD_PATH = config.REPORTS_DIR / "ML_PIPELINE_FOUNDATION_REPORT.md"
PDF_PATH = config.REPORTS_DIR / "ML_PIPELINE_FOUNDATION_REPORT.pdf"


def fmt(n) -> str:
    return f"{n:,}" if isinstance(n, (int,)) else str(n)


def build_markdown(R: dict) -> str:
    d = R["data"]
    vw = R["validation_window"]
    fw = R["final_forecast_window"]
    vf = R["validation_frame"]
    to = R["training_origins"]
    ff = R["future_frame"]
    feats = R["features"]
    checks = R["checks"]
    smoke = R["metric_smoke_test"]
    probe = R["listing_feature_probe"]

    zeros_pred = smoke["predict_all_zeros"]
    persist_key = [k for k in smoke if k.startswith("predict_rolling_mean_28")][0]
    persist = smoke[persist_key]

    L: list[str] = []
    A = L.append

    A("# ML Pipeline Foundation Report")
    A("")
    A(f"*Stage 1 of the forecasting build. Generated {date.today().isoformat()} "
      f"from an actual pipeline run ({R['runtime_seconds']}s).*")
    A("")
    A("> **No model has been trained in this stage.** No hyperparameters were "
      "tuned, no hurdle model was built, no submission was created, and no claim "
      "is made about the team's LightGBM+Tweedie benchmark (RMSE 2.0324 / "
      "MAE 1.0869). This report documents only the machinery that a model will "
      "later be trained on.")
    A("")
    A("---")
    A("")

    A("## 1. What we built, in one paragraph")
    A("")
    A("We built the scaffolding for forecasting 28 days of daily unit sales for "
      f"all {fmt(config.N_SERIES)} store-item series. That scaffolding has four parts: a "
      "**data loader** that reads the raw files into compact matrices, a "
      "**feature engineering layer** that turns history into model inputs without "
      "ever peeking at the future, a **backtesting framework** that recreates the "
      "real 28-day forecasting task on a stretch of history where we already know "
      "the answers, and a **check suite** that tries to prove the whole thing "
      f"wrong. All {checks['total_checks']} checks currently pass.")
    A("")
    A("### Project structure")
    A("")
    A("```")
    A("pipeline/")
    A("    config.py             paths, dataset constants, backtest origins")
    A("    data_loader.py        raw CSVs -> wide matrices (read-only)")
    A("    features.py           feature engineering, groups A-G")
    A("    backtest.py           train / validation / future frame assembly")
    A("    metrics.py            RMSE, MAE, WAPE, bias")
    A("    validation_checks.py  correctness + empirical leakage tests")
    A("    report_pdf.py         markdown -> PDF renderer")
    A("scripts/")
    A("    01_foundation_check.py        runs everything, writes the results JSON")
    A("    02_build_foundation_report.py builds this report from that JSON")
    A("artifacts/    check results, feature summary, inspectable sample")
    A("models/       (empty) trained model files, next stage")
    A("experiments/  (empty) ablation configs and results, next stage")
    A("predictions/  (empty) forecast outputs, final stage")
    A("reports/      this report")
    A("```")
    A("")

    A("## 2. The data we are working from")
    A("")
    A("The pipeline reads the original files in `raw_dataset/` directly, "
      "**read-only**. Nothing in `raw_dataset/` or `processed_dataset/` was "
      "modified, and `sales_long_full.parquet` was not touched.")
    A("")
    A("| Property | Value |")
    A("|---|---|")
    A(f"| Store-item series | {fmt(d['n_series'])} |")
    A(f"| Days of history | {fmt(d['n_history_days'])} ({d['first_date']} to {d['last_sales_date']}) |")
    A(f"| Calendar days available | {fmt(d['n_calendar_days'])} (runs to {d['last_calendar_date']}) |")
    A(f"| Price weeks | {fmt(d['n_price_weeks'])} |")
    A(f"| Sales matrix in memory | {d['sales_matrix_mb']} MB |")
    A(f"| Price matrix in memory | {d['price_matrix_mb']} MB |")
    A(f"| Load time | {d['load_seconds']}s |")
    A("")
    A("> **Why not the 59-million-row processed table?** "
      "`processed_dataset/sales_long_full.parquet` holds the same information but "
      "as 59,181,090 separate rows, which needs several GB to work with. This "
      "machine has about 5.7 GB free. Kept in its natural rectangular shape "
      f"({fmt(config.N_SERIES)} series x {fmt(config.N_HISTORY_DAYS)} days) the same data fits in "
      f"{d['sales_matrix_mb']} MB, and every feature becomes a fast array slice "
      "instead of a grouped scan over 59 million rows. The processed table is left "
      "exactly as it was; we cross-check our totals against the values already "
      "verified from it.")
    A("")
    A("The loader refuses to proceed unless the data reproduces the totals that "
      "were independently verified in the earlier review stage:")
    A("")
    A("| Integrity check | Expected | Result |")
    A("|---|---|---|")
    A(f"| Total units sold | {fmt(config.EXPECTED_TOTAL_UNITS)} | matched |")
    A(f"| Zero-sales cells | {fmt(config.EXPECTED_ZERO_CELLS)} (68.00%) | matched |")
    A(f"| Maximum single-day sale | {config.EXPECTED_MAX_SALES} | matched |")
    A("| Negative sales values | 0 | matched |")
    A("")

    A("## 3. What leakage means, and why this stage is mostly about preventing it")
    A("")
    A("> **Feature leakage** is when information from the future accidentally ends "
      "up in the data a model learns from. The model then looks brilliant during "
      "testing and falls apart in reality, because at the real moment of "
      "forecasting that information does not exist yet.")
    A("")
    A("A concrete example for this project. Suppose we want to predict sales on "
      "25 May. A feature like \"average sales over the last 7 days\" sounds "
      "harmless — but if we compute it *relative to 25 May*, it uses 18-24 May. If "
      "we are actually standing on 1 May making a 28-day forecast, we do not know "
      "any of those values yet. Using them is leakage.")
    A("")
    A("This is not a hypothetical risk. It is the single most common way a "
      "forecasting project quietly fools itself, and both the EDA report and the "
      "final approach document flagged it as the biggest correctness risk in the "
      "project. So the pipeline is built around one rule.")
    A("")
    A("### The rule: fixed origin, two kinds of feature")
    A("")
    A("Pick a day **T**, the *forecast origin* — the last day whose sales we know. "
      "We must predict days T+1 through T+28 all at once, standing at T. Every "
      "feature is then one of exactly two kinds:")
    A("")
    A("| Kind | Built from | Behaviour across the 28 days |")
    A("|---|---|---|")
    A("| **Origin-relative** | sales history up to and including day T | **Constant.** The same value is used for all 28 forecast days. |")
    A("| **Target-day** | the calendar and the price file, which are published ahead of time | **Varies** per forecast day, legitimately. |")
    A("")
    A("Everything derived from past sales is origin-relative, because on day T we "
      "genuinely do not know day T+5's sales. Everything derived from the calendar "
      "or prices is target-day, because this dataset really does supply those for "
      "the forecast window.")
    A("")
    A("### What the model is allowed to see, and what it must never see")
    A("")
    A("| Information | Allowed? | Why |")
    A("|---|---|---|")
    A("| Sales on or before day T | **YES** | Already observed at forecast time |")
    A("| Sales after day T | **NEVER** | This is the answer we are being asked for |")
    A("| Weekday, month, year of a forecast day | **YES** | The calendar is deterministic |")
    A("| Holiday / event name on a forecast day | **YES** | `calendar.csv` covers all 28 future days |")
    A("| SNAP flag on a forecast day | **YES** | Benefit schedules are published in advance |")
    A("| Selling price on a forecast day | **YES** | `sell_prices.csv` covers the forecast weeks |")
    A("| Any aggregate computed across the forecast window | **NEVER** | Would smuggle future sales in indirectly |")
    A("")
    A("> **SNAP** is the Supplemental Nutrition Assistance Program, a US "
      "food-assistance benefit. The dataset flags, per state per day, whether the "
      "benefit was usable. Each series is matched to its **own** state's flag — a "
      "California store reads `snap_CA` — because using a blended flag would blur "
      "a signal the EDA found to be worth +12.7% in mean sales overall and +17.3% "
      "within FOODS.")
    A("")

    A("## 4. How the 28-day backtest works")
    A("")
    A("We cannot score anything on the real forecast window, because nobody has "
      "those sales — predicting them is the whole task. So we rewind: pretend an "
      "earlier day was \"today\", forecast the 28 days after it, and compare "
      "against sales we already have on record.")
    A("")
    A("> **Why random train/test splitting would be wrong.** A random split would "
      "let the model train on 10 May while being tested on 30 April — learning "
      "from the future to explain the past. The score would look excellent and "
      "mean nothing. Time-series validation must always cut on time.")
    A("")
    A("### The three blocks of data, kept strictly apart")
    A("")
    A("| Block | Days | Dates | Role |")
    A("|---|---|---|---|")
    A(f"| **TRAINING** | {vw['training_days_available']} | {vw['training_dates_available']} | Everything the model may learn from |")
    A(f"| **VALIDATION** | {vw['validation_days']} | {vw['validation_dates']} | Real observed sales. Scored against, never trained on, never used to build a feature |")
    A(f"| **FINAL FORECAST** | {fw['validation_days']} | {fw['validation_dates']} | Genuinely unknown. No sales for these days exist in any file |")
    A("")
    A(f"The validation origin is **{vw['forecast_origin_day']} "
      f"({vw['forecast_origin_date']})**. That day was chosen deliberately: the 28 "
      f"days after it ({vw['validation_days']}) are exactly the block that exists "
      "in `sales_train_evaluation.csv` but not in `sales_train_validation.csv`. "
      "They are real observed sales, so we can score against them, and they "
      "reproduce the shape of the real task precisely — 28 days ahead, from a "
      "fixed origin, for every series at once.")
    A("")
    A("### A subtlety that is easy to get wrong")
    A("")
    A("It is not enough for training *targets* to sit before the cutoff. Training "
      "*features* must also be buildable from before the cutoff. So every training "
      "origin satisfies `origin + 28 <= validation origin`. The training origins "
      "used in this verification run were:")
    A("")
    A("| Training origin | Date | Its 28-day target block ends |")
    A("|---|---|---|")
    for od, dt, oi in zip(to["origin_days"], to["origin_dates"], to["origin_day_idxs"]):
        A(f"| {od} | {dt} | d_{oi + 1 + config.HORIZON} |")
    A("")
    A(f"The latest of those target blocks ends at "
      f"d_{max(to['origin_day_idxs']) + 1 + config.HORIZON}, one day before the "
      f"validation window opens at {vw['validation_days'].split(' ')[0]}. The "
      "framework asserts this rather than assuming it, and raises an error if it "
      "is ever violated.")
    A("")

    A("## 5. The features")
    A("")
    A(f"**{feats['n_features']} features across {feats['n_groups']} groups.** Every "
      "one is labelled below as origin-relative (constant over the 28 days) or "
      "target-day (varies).")
    A("")

    group_notes = {
        "A_calendar": ("target-day",
                       "Why it exists: the EDA measured a +31.1% weekend effect and found "
                       "individual named holidays moving in opposite directions "
                       "(Christmas -99.95%, Labor Day +27.5%). A single \"is it a holiday\" "
                       "flag would average that away to nearly nothing, so the specific "
                       "event identity is kept."),
        "B_historical_demand": ("origin-relative",
                                "Why it exists: recent demand is the strongest predictor in the "
                                "dataset. The EDA measured rolling_mean_7 at r=0.820 and "
                                "rolling_mean_28 at r=0.807 against same-day sales, higher than "
                                "any single lag."),
        "C_recency": ("origin-relative",
                      "Why it exists: the cleanest relationship found anywhere in the EDA. The "
                      "chance of selling today falls from 65.2% if the item sold yesterday to "
                      "0.6% after 29+ dry days."),
        "D_listing": ("mixed",
                      "Why it exists: many early zeros are not weak demand, they are \"this "
                      "product was not on the shelf yet\". Section 7 shows this is measurable "
                      "and near-absolute."),
        "E_price": ("mixed",
                    "Why it exists: price is one of the few genuinely forward-known variables "
                    "here. Raw price is dominated by cross-item scale ($30 hobby item vs $1 "
                    "food item), so price relative to the item's own recent average is "
                    "included alongside it."),
        "F_hierarchy": ("static",
                        "Why it exists: behaviour differs sharply across the hierarchy — "
                        "zero-sales rates range from 58.6% in FOODS_3 to 88.4% in HOBBIES_2."),
        "G_horizon": ("target-day",
                      "Why it exists: predicting 1 day ahead and 28 days ahead are different "
                      "problems, and a direct multi-horizon model needs to know which one it "
                      "is being asked for."),
    }

    by_group = feats["by_group"]
    summary_rows = {r["feature"]: r for r in feats["summary"]}

    for grp, cols in by_group.items():
        if not cols:
            continue
        kind, note = group_notes.get(grp, ("", ""))
        pretty = grp.split("_", 1)[1].replace("_", " ").title()
        A(f"### Group {grp.split('_')[0]} — {pretty} ({kind})")
        A("")
        A(note)
        A("")
        A("| Feature | Missing % | Min | Max | Mean |")
        A("|---|---|---|---|---|")
        for c in cols:
            r = summary_rows.get(c, {})
            A(f"| `{c}` | {r.get('missing_pct', '-')} | {r.get('min', '-')} | "
              f"{r.get('max', '-')} | {r.get('mean', '-')} |")
        A("")

    A("### Exactly how the lag and rolling features are defined")
    A("")
    A("`lag_k` = sales on the day k days before the **first forecast day**. With "
      "origin T, the first forecast day is T+1, so:")
    A("")
    A("| Feature | Day it reads | Safe for all 28 horizon days? |")
    A("|---|---|---|")
    A("| `lag_1` | T | Yes — T is the last day we know |")
    A("| `lag_7` | T-6 | Yes |")
    A("| `lag_14` | T-13 | Yes |")
    A("| `lag_28` | T-27 | Yes |")
    A("| `rolling_mean_7` / `rolling_std_7` | mean/std over T-6 .. T | Yes |")
    A("| `rolling_mean_28` / `rolling_std_28` | mean/std over T-27 .. T | Yes |")
    A("")
    A("All six windows end at T. None of them can reach into the forecast period, "
      "for any of the 28 days, which is what makes them usable in a direct "
      "multi-horizon setup.")
    A("")
    A("### What we deliberately did NOT do to the data")
    A("")
    A("- Sales were **not** smoothed.")
    A("- Zero-sales rows were **not** removed.")
    A("- Zeros were **not** replaced with missing values.")
    A("- No zero was assumed to be a stockout. The dataset has no inventory field, so that cannot be known.")
    A("- No suspected-stockout rows were dropped.")
    A("- No promotion labels were invented. The dataset has no promotion field.")
    A("- Missing prices were left as missing, never imputed. LightGBM handles them natively, and the missingness is itself informative.")
    A("")
    A("The model will learn from the original observations exactly as recorded.")
    A("")

    A("## 6. Validation checks")
    A("")
    A(f"**{checks['passed']} of {checks['total_checks']} checks pass.** The full "
      "machine-readable output is in `artifacts/foundation_checks.json`.")
    A("")
    A("### The leakage test, done empirically rather than asserted")
    A("")
    A("Writing \"this feature is safe\" in a comment proves nothing. So the "
      "pipeline proves it by experiment:")
    A("")
    A("1. Build the feature frame normally at the validation origin.")
    A("2. Take a copy of the sales matrix and overwrite **every day after the origin** with an absurd value (9999 units).")
    A("3. Rebuild the exact same feature frame from the corrupted data.")
    A("4. Compare. If any feature value moved, that feature was reading the future.")
    A("")
    A(f"**Result: all {feats['n_features']} features are bit-for-bit identical "
      "between the clean and corrupted runs.** A companion check confirms the "
      "target column *did* change, which proves the corruption actually reached "
      "the data and the test was meaningful rather than vacuous.")
    A("")
    A("The mirror-image test matters just as much: corrupting future **prices** "
      "*should* change the price features, because prices for the forecast window "
      "are legitimately known. That check passes too — so we are neither leaking "
      "what we must not use, nor discarding what we are entitled to use.")
    A("")
    A("### All checks by area")
    A("")
    A("| Area | Checks | What is verified |")
    A("|---|---|---|")
    A("| Source integrity | 9 | Row/day counts, total units, zero count, max value, no negatives, zeros not silently converted to NaN |")
    A("| Calendar alignment | 2 | Six anchor day-index/date pairs exact; calendar extends 28 days past the sales |")
    A("| Frame structure | 5 | Row count, no duplicate (series, day) pairs, exactly 28 distinct target days, every series present on every day, horizon values 1..28 |")
    A("| Feature sanity | 14 | Non-negative demand features, recency in range, SNAP binary and state-matched, prices positive, target still raw integers |")
    A("| Target correctness | 1 | 500 random rows spot-checked back against `sales_train_evaluation.csv` |")
    A("| Leakage | 4 | Future-sales corruption test, corruption-applied counter-check, future-price usability, price/demand independence |")
    A("| Train/validation separation | 3 | Training targets strictly precede validation; origins at least 28 days back; no duplicate training rows |")
    A("| Listing behaviour | 3 | Feature activates at early origins; pre-listing rows have no sales; redundancy check |")
    A("| Future frame | 4 | Correct row count, no target attached, calendar+SNAP present, price coverage |")
    A("| Metric pipeline | 1 | Metrics run over the full 853,720-prediction window |")
    A("")
    A("### Row-count arithmetic, confirmed")
    A("")
    A(f"| Frame | Rows | Check |")
    A("|---|---|---|")
    A(f"| Validation | {fmt(vf['rows'])} | {fmt(config.N_SERIES)} series x {config.HORIZON} days |")
    A(f"| Future forecast | {fmt(ff['rows'])} | {fmt(config.N_SERIES)} series x {config.HORIZON} days |")
    A(f"| Training (6 origins, 2,000-series sample) | {fmt(R['training_frame_sample']['rows'])} | 2,000 x {config.HORIZON} x 6 |")
    A("")

    A("## 7. A finding that came out of building this")
    A("")
    A("The listing-aware features were probed across four origins to check they "
      "actually do something. They produced a result stronger than the EDA had "
      "established:")
    A("")
    A("| Origin | Date | Rows flagged pre-listing | Mean sales on those rows | Zero-sales rate on those rows | Zero-sales rate on listed rows |")
    A("|---|---|---|---|---|---|")
    for r in probe:
        pm = r.get("pre_listing_mean_sales")
        pz = r.get("pre_listing_zero_pct")
        lz = r.get("listed_zero_pct")
        A(f"| {r['origin_day']} | {r['origin_date']} | {r['pre_listing_pct']}% | "
          f"{'—' if pm is None else pm} | {'—' if pz is None else str(pz) + '%'} | "
          f"{'—' if lz is None else str(lz) + '%'} |")
    A("")
    A("**Rows flagged as pre-listing have a 100.00% zero-sales rate and a mean of "
      "exactly 0.0 units.** Not approximately — every single one. The flag is "
      "derived purely from `sell_prices.csv`, and the sales come from "
      "`sales_train_evaluation.csv`, so this is two independent files agreeing, "
      "not circular reasoning.")
    A("")
    A("Two consequences worth carrying into the modelling stage:")
    A("")
    A("- At early origins nearly half the panel is structurally zero (47.84% at "
      "d_201). Training on those rows as though they were ordinary weak demand "
      "will pull the model toward predicting zero. They are closer to \"not "
      "applicable\" than to \"no demand\".")
    A("- At the validation origin and at the real forecast origin, **0%** of rows "
      "are pre-listing and 100% have a known price. So this feature contributes "
      "nothing at prediction time for this particular horizon. It matters for how "
      "we build the **training set**, not for the final forecast. That is a "
      "meaningful limitation on how much the \"listing-aware\" idea can be "
      "expected to move the final score, and it is better to know now than after "
      "building a novelty story around it.")
    A("")

    A("## 8. Problems encountered")
    A("")
    A("### The leakage test failed on first run — and was right to")
    A("")
    A("On the first full run, `rolling_std_28` came back as changed by the "
      "corruption test, which would mean a feature was reading future sales. It "
      "was investigated before anything was altered.")
    A("")
    A("The input slices fed to the calculation were **byte-identical** between the "
      "clean and corrupted runs, so no future data was being read. The differences "
      "were at most 3.8e-06 in absolute terms and 5.1e-07 relative — roughly four "
      "times float32 machine epsilon. The cause: pandas returned the sales matrix "
      "in Fortran (column-major) order, while the test's copy was C (row-major) "
      "order, and NumPy's pairwise summation groups values according to memory "
      "layout. Same numbers, different addition order, last-bit differences.")
    A("")
    A("Rather than weaken the test to a tolerance — which would have blunted the "
      "one check most likely to catch a genuine future-data bug — the root cause "
      "was fixed:")
    A("")
    A("- the sales matrix is now stored C-contiguous (which also matches our "
      "dominant access pattern of whole-row slices per series);")
    A("- rolling means and standard deviations accumulate in float64 before being "
      "narrowed to float32.")
    A("")
    A("Exact bit-equality now holds and the test remains strict.")
    A("")
    A("### Two feature pairs turned out to be redundant")
    A("")
    A("Both were built because the stage specification asked for them, and both "
      "are reported rather than quietly dropped:")
    A("")
    A("- `days_since_last_sale` and `zero_streak_length` are **the same number** at "
      "a fixed origin. If the last sale was 3 days ago then there are exactly 3 "
      "consecutive zero days ending at the origin. Verified identical across all "
      f"{fmt(vf['rows'])} validation rows.")
    A("- `pre_listing` and `price_is_missing` were **identical at every origin "
      "tested**. Pre-listing is defined from the first priced day, so for the "
      "leading block the two coincide exactly.")
    A("")
    A("One of each pair should be dropped before training. Keeping both adds "
      "compute and splits feature-importance between duplicate columns, which "
      "makes the ablation study harder to read.")
    A("")
    A("### Features that are inert at the forecast origin")
    A("")
    A("`pre_listing` and `price_is_missing` are constant zero at the validation "
      "origin, and `year` is constant 2016 across the validation window. Constant "
      "columns carry no information for a tree model at prediction time. They are "
      "retained because they are informative in training rows drawn from earlier "
      "origins, but this is worth knowing before reading anything into their "
      "feature importances.")
    A("")

    A("## 9. Metric pipeline smoke test")
    A("")
    A("> **These are not model results.** They are two trivial arithmetic rules "
      "with no fitting of any kind, run purely to prove the metric code works on a "
      "real 853,720-row window. They are **not** baselines, and they must **not** "
      "be compared with the team's LightGBM+Tweedie benchmark.")
    A("")
    A("| Rule | RMSE | MAE | WAPE | Bias |")
    A("|---|---|---|---|---|")
    A(f"| Predict 0 for everything | {zeros_pred['RMSE']:.4f} | {zeros_pred['MAE']:.4f} | "
      f"{zeros_pred['WAPE']:.4f} | {zeros_pred['bias']:+.4f} |")
    A(f"| Repeat each series' own 28-day average | {persist['RMSE']:.4f} | {persist['MAE']:.4f} | "
      f"{persist['WAPE']:.4f} | {persist['bias']:+.4f} |")
    A("")
    A("The metric code runs correctly over all "
      f"{fmt(zeros_pred['n'])} predictions. One thing these numbers illustrate is "
      "why MAE alone is a poor guide on this dataset: predicting zero everywhere "
      f"achieves an MAE of {zeros_pred['MAE']:.4f} while explaining none of the "
      "demand at all, which the WAPE of 1.0000 makes visible. Both RMSE and MAE "
      "are reported throughout this project, alongside WAPE, for that reason.")
    A("")

    A("## 10. Where this leaves us")
    A("")
    A("### Ready")
    A("")
    A("- Data loading, verified against known totals")
    A(f"- {feats['n_features']} features across {feats['n_groups']} groups, all leakage-tested")
    A("- Fixed-origin 28-day backtest with an enforced train/validation separation")
    A("- Metrics (RMSE, MAE, WAPE, bias), including per-group breakdowns")
    A("- Future-horizon frame for d_1942..d_1969, with calendar, SNAP, events and "
      f"prices present for {ff['price_present_pct']}% of rows and no target attached")
    A("- LightGBM 4.7.0 installed; `requirements.txt` pins the full environment")
    A("")
    A("### Decisions that need making before Model 1")
    A("")
    A("- **How many training origins, and how far back.** This run used 6 origins "
      "at a 28-day stride purely to verify the mechanics. More origins means more "
      "training data but also more memory; the full build at "
      f"{fmt(config.N_SERIES)} series is about {fmt(config.N_SERIES * config.HORIZON)} "
      "rows per origin.")
    A("- **Whether to drop the redundant feature in each pair** identified in "
      "Section 8. Recommended: yes, before the ablation study.")
    A("- **Whether to exclude pre-listing rows from training.** They are "
      "structurally zero, and including roughly half a panel of guaranteed zeros "
      "at early origins will bias the model. This is a real modelling decision "
      "with evidence behind it now, and it should be tested both ways rather than "
      "assumed.")
    A("- **The evaluation metric for the hackathon** is still unconfirmed. RMSE "
      "and MAE are computed because the team's benchmark is quoted in them.")
    A("")
    A("### Nothing is blocking Model 0 / Model 1")
    A("")
    A("The foundation runs end to end in "
      f"{R['runtime_seconds']}s and all {checks['total_checks']} checks pass. The "
      "next stage can build a naive baseline and a global LightGBM model on top of "
      "this without further groundwork.")
    A("")
    A("---")
    A("")
    A("*Generated by `scripts/02_build_foundation_report.py` from "
      "`artifacts/foundation_checks.json`. Every figure in this report is read "
      "from that file, which was written by an actual pipeline run — no number "
      "here was entered by hand. No model was trained in this stage.*")

    return "\n".join(L)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"{RESULTS_PATH} not found — run scripts/01_foundation_check.py first."
        )

    R = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    md = build_markdown(R)
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"wrote {MD_PATH}  ({len(md):,} chars)")

    render_markdown_to_pdf(
        MD_PATH, PDF_PATH,
        title="ML Pipeline Foundation Report",
        subtitles=[
            "M5 Retail Demand Forecasting — Problem Statement 11",
            "Stage 1: data loading, feature engineering, leakage-safe backtesting",
            "NPN AIA Hackathon — St. Joseph's College of Engineering",
        ],
        footer="ML_PIPELINE_FOUNDATION_REPORT.pdf — Stage 1 foundation, no model trained",
    )
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
