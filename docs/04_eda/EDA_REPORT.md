# M5 Sales Data — Exploratory Data Analysis & Feature-Design Investigation

**Project:** 28-day sales forecasting hackathon
**Scope of this stage:** Understand what the dataset contains, what patterns exist in demand, and what information a future forecasting model might use. **No model was trained. No algorithm or novelty was chosen.** Those decisions are for the team.

---

## 1. Objective

Answer three questions before the team commits to a modeling strategy: **What does the dataset actually contain? What patterns and problems exist in demand? What information should we consider giving a future forecasting model?** Every claim below is backed by a statistic, table, or chart generated directly from the data — see `EDA_METHODOLOGY.md` for exactly how, and `EDA/statistics/`, `EDA/tables/`, `EDA/charts/` for the underlying evidence.

## 2. Dataset Used

**Source:** `processed_dataset/sales_long_full.parquet` — the validated base dataset built and documented in `processed_dataset/PROCESSING_REPORT.md`. This EDA reads that file only; it never touches `raw_dataset/` and never writes to the Parquet file itself.

## 3. Dataset Structure (Phase 1)

| Metric | Value |
|---|---|
| Total rows | 59,181,090 |
| Total columns | 22 |
| Unique series (item x store) | 30,490 |
| Unique items | 3,049 |
| Unique stores | 10 |
| Unique departments | 7 |
| Unique categories | 3 |
| Unique states | 3 |
| Date range | 2011-01-29 to 2016-05-22 |
| Number of days | 1,941 |
| Duplicate (id, date) pairs | 0 |
| Missing values (id/date/calendar/sales columns) | 0 |
| Missing `event_name_1` / `event_type_1` | 54,363,670 (91.86%) — expected, most days have no primary event |
| Missing `event_name_2` / `event_type_2` | 59,059,130 (99.79%) — expected, secondary events are rare |
| Missing `sell_price` | 12,299,413 (20.78%) — expected, see Section 10 |
| Sales min / max / mean / median | 0 / 763 / 1.131 / 0.0 |
| Total sales (units) | 66,927,173 |
| Zero-sales rows | 40,241,819 (68.00%) |
| Non-zero-sales rows | 18,939,271 (32.00%) |

Full detail: `EDA/statistics/phase1_sanity_check.json`. These figures match `processed_dataset/PROCESSING_REPORT.md` exactly, confirming the dataset has not drifted since that stage.

## 4. EDA Methodology

Summarized here; full detail in `EDA_METHODOLOGY.md`. In short: every phase runs directly against the full 59.18M-row table (one exception, documented below), time-order-dependent computations explicitly sort by `(id, date)` first (the file's physical storage order is date-major, not id-major), zero values are never altered, and activity-class thresholds are derived from this dataset's own distribution rather than picked in advance.

**The one sampling exception:** Phase 8's per-series event/price sensitivity uses a stratified random sample (seed=99, n=501 of 30,490 series, proportional across category x volume-tier). Everything else — including all lag/rolling correlations and the zero-streak analysis — runs on the complete dataset.

## 5. Sales Distribution Findings (Phase 2)

**What was analyzed:** the shape of the sales value distribution, mean vs. median, percentiles, and how total sales are spread across categories, departments, stores, states, and individual series.

**Why:** a forecasting approach needs to match the actual shape of the target — a heavily zero-inflated, right-skewed count variable behaves very differently from a smooth continuous quantity.

**What was found:**

- The distribution is extremely right-skewed: **skewness = 17.1** across all rows (11.9 even after dropping zeros). 75% of all rows are 0 or 1 unit; the p99 is only 15 units, but the max is 763.
- Mean (1.131) is far above the median (0.0) — a classic signature of a zero-inflated, long-tailed count variable.
- Sales concentrate heavily in a minority of series: the **top 10% of series by total sales account for 54.4% of all units sold**, while the **bottom 10% account for only 0.52%** (`EDA/charts/05_sales_concentration_lorenz.png`).
- Category matters enormously: **FOODS accounts for 68.6% of total sales** despite being 1 of 3 categories, and within it, department **FOODS_3 alone is 49.2% of total company-wide sales**. HOBBIES is the smallest and sparsest category (77.1% zero rows vs. 61.8% for FOODS).
- Store-level total-sales share ranges from 6.2% (CA_4) to 17.0% (CA_3) — a ~2.7x spread across otherwise-comparable stores (`EDA/tables/sales_by_store.csv`).
- 51.4% of series have a coefficient of variation (std/mean) above 2 — more than half the catalog is highly volatile relative to its own average.

**Why it matters:** a single global model is very likely to be dominated by FOODS/FOODS_3 unless the loss function or sampling strategy accounts for category imbalance. The heavy concentration in a few high-volume series also means aggregate accuracy metrics can look good while badly under-serving the long tail of low-volume items.

Charts: `01_sales_value_frequency.png`, `02_zero_vs_nonzero_share.png`, `03_sales_mean_median_by_category.png`, `04_sales_share_by_store.png`, `05_sales_concentration_lorenz.png`. Tables: `sales_by_category.csv`, `sales_by_department.csv`, `sales_by_store.csv`, `sales_by_state.csv`, `series_level_sales_stats.csv`.

## 6. Zero-Sales / Intermittent Demand Findings (Phase 3) — HIGH PRIORITY

**What was analyzed:** overall and grouped zero-sales rates, the distribution of zero% across all 30,490 series, consecutive zero-run ("streak") lengths, and whether long *leading* zero blocks (before a series' first sale) line up with the period before that series had any recorded price.

**Why:** with 68% of rows at zero, how those zeros are distributed and structured matters more for modeling strategy than almost anything else in this dataset.

**What was found:**

- Zero% varies substantially by category (HOBBIES 77.1% vs. FOODS 61.8%) and by store (CA_3 59.4% vs. CA_4 72.0%) — not a uniform property of the dataset.
- Across the 30,490 series, per-series zero% itself has a wide distribution: p5 = 24.7%, p50 = 73.3%, p95 = 95.0%. Using these quartiles as data-driven (not arbitrary) cutoffs: **7,629 series are "High-activity"** (zero% <= 53.5%), **15,267 "Regular/intermittent"**, **6,103 "Sparse"**, and **1,491 "Extremely sparse"** (zero% > 95.0%). No series is 100% zero.
- Consecutive zero-sales streaks: median 2 days, mean 6.08 days, p90 = 7 days, p99 = 52 days, max = 1,845 days (a series essentially dormant for most of its history). 44% of all zero-runs are just 1-3 days long — most "dry spells" are short.
- **Leading zero blocks are, to a striking degree, explained by pre-listing timing, not random intermittency.** For series with a leading zero block, the gap until the first non-zero sale and the gap until the first available price line up almost exactly: **median difference of only 3 days, and 99.48% of series are within 7 days of each other.** Only 0.29% of series show a leading-zero gap that exceeds the leading-no-price gap by more than 30 days.

**Fact vs. interpretation, explicitly separated:**
- **FACT:** leading zero-sales periods and leading no-price periods end at almost the same time for the vast majority of series.
- **INTERPRETATION (hypothesis, well-supported):** this pattern is consistent with items simply not yet being listed/available at a store early in their observed window, rather than being listed-but-not-selling. This is *not* a claim about stockouts — the dataset has no inventory field, and this analysis only speaks to the *very first* zero block of each series, not zeros that occur later after a series is clearly active.
- Later, non-leading zero runs are a separate phenomenon (ordinary intermittent demand) and this analysis does not attempt to label their cause.

**Why it matters:** a naive model that treats every zero identically will conflate "hasn't been listed yet" with "listed but no demand today" — two very different situations with very different modeling implications (the first is closer to missing data than a demand signal).

Charts: `06_series_zero_pct_distribution.png`, `07_zero_pct_by_category_store.png`, `08_zero_streak_length_distribution.png`, `09_leading_zero_vs_noprice_gap.png`. Tables: `zero_pct_by_category.csv`, `zero_pct_by_department.csv`, `zero_pct_by_store.csv`, `series_zero_pct.csv`, `series_classification_by_category.csv`, `zero_streak_length_distribution.csv`, `zero_streak_by_category.csv`, `zero_streak_by_department.csv`, `zero_streak_by_store.csv`, `series_leading_zero_and_price_gap.csv`.

## 7. Temporal / Seasonality Findings (Phase 4)

**What was analyzed:** day-of-week, monthly, yearly, and week-of-year sales patterns, plus a check for whether the apparent yearly growth trend is genuine or a side-effect of more items being listed over time.

**Why:** these are the most basic candidate signals for any time-series model, and the dataset's rectangular structure (every series has a row for every day, even before it was actually stocked) creates a specific risk of mistaking assortment growth for organic demand growth.

**What was found:**

- **Weekends are meaningfully higher:** mean sales on Saturday/Sunday (1.361) are **31.1% higher** than weekday-only mean sales (1.038), and this holds within every category (FOODS Sat/Sun ~1.97 vs. weekday ~1.5; similar pattern in HOBBIES and HOUSEHOLD).
- **Monthly effects are real but modest:** August is highest (104.3% of the overall mean), December is lowest (95.6%) — an 8.99% high-low spread, much smaller than the weekend effect.
- **Week-of-year effects** show a similar modest range (week 36 highest at 1.231, week 52 lowest at 0.944 — a late-December dip, consistent with the Christmas closure finding in Section 8).
- **The apparent yearly sales growth (2011→2015) is at least partly a composition effect, not proven organic growth.** Total sales rose from 8.86M (2011) to 13.80M (2015), but the cumulative count of series with an observed price rose from 16,762 to 30,474 over the same period — the catalog nearly doubled in the same window the "trend" appears in (`EDA/charts/13_yearly_sales_vs_active_series.png`). **2016 is a partial year** (data ends 2016-05-22), so its lower total is an artifact of coverage, not declining demand.

**Why it matters:** any model or feature (e.g., "year" or a raw trend term) that doesn't account for the assortment-growth confound risks learning "more items on shelves" rather than genuine seasonal or secular demand growth.

Charts: `10_mean_sales_by_weekday.png`, `11_mean_sales_by_month_category.png`, `12_daily_total_sales_trend.png`, `13_yearly_sales_vs_active_series.png`, `14_mean_sales_by_week_of_year.png`. Tables: `mean_sales_by_weekday_and_category.csv`, `mean_sales_by_month_and_category.csv`, `yearly_sales_stats.csv`, `daily_total_sales.csv`.

## 8. Event / Holiday Findings (Phase 5)

**What was analyzed:** sales on event vs. non-event days overall, by event type, and by category/department/store; and, for each of the 30 distinct named events with 2+ occurrences, sales on the event date compared to a **local baseline** (same weekday, non-event dates within +/-21 days) — controlling for the weekday/seasonal effects found in Section 7 rather than comparing against the dataset's flat overall average.

**Why:** the calendar is known well in advance for the entire 28-day forecast horizon, so any real event effect is a genuinely usable predictive signal, not just a descriptive curiosity.

**What was found:**

- **In aggregate, event days look unremarkable or even slightly lower** (event-day mean sales are 4.6% *lower* than non-event days overall, and lower in every category). This aggregate number is misleading — it averages together events with opposite effects.
- **Individual named events have large, specific, and very different effects:**
  - **Christmas: -99.95%** vs. local baseline — sales are essentially zero, consistent with stores being closed.
  - Thanksgiving: -29.9%. NewYear: -20.4%. Mother's day: -12.5%. Easter: -11.4%. Halloween: -10.0%.
  - LaborDay: **+27.5%**. ColumbusDay: +9.6%. VeteransDay: +8.8%. Cinco De Mayo: +7.1%.
- Event effects also vary by store (WI_2 shows the largest event-day drop at -9.1%, TX_1 is roughly flat at +0.6%) and by category (HOBBIES most negative on event days at -8.7%, FOODS least at -3.2%).

**Why it matters:** a single "is_event" binary flag would badly underuse this signal — Christmas and LaborDay point in opposite directions with a magnitude gap of over 100 percentage points. The specific event identity, not just its presence, carries the useful information. No event here is labeled a "promotion" — the dataset has no such field, and these are calendar effects (store closures, gift-shopping patterns), not confirmed promotional activity.

Charts: `15_event_effect_by_category.png`, `16_named_event_lift.png`. Tables: `event_effect_by_category.csv`, `event_effect_by_department.csv`, `event_effect_by_store.csv`, `event_name_aggregated_lift.csv`, `event_occurrence_level_lift.csv`.

## 9. SNAP Findings (Phase 6)

**What was analyzed:** sales on SNAP vs. non-SNAP days, using each row's *own state's* SNAP flag (`snap_CA`/`snap_TX`/`snap_WI` matched to `state_id`) — not a single shared flag — broken down by state, category, department, and store.

**Why:** SNAP (food-assistance) benefit distribution days are published on a fixed calendar and represent a genuine, forward-known candidate signal specifically tied to food spending.

**What was found:**

- SNAP days show **12.7% higher mean sales** overall than non-SNAP days.
- The effect is strongly concentrated in food: **FOODS +17.3%**, vs. HOUSEHOLD +3.5% and HOBBIES +2.5%. Within FOODS, department **FOODS_2 shows a +32.3% lift** — the largest of any department.
- The effect holds directionally in all three states but varies in size: WI +21.8%, TX +11.5%, CA +8.0%.
- ~33% of days are SNAP days in each state (`pct_of_days_that_are_snap_days_per_state` in `phase5_6_events_snap.json`).

**Why it matters:** this is one of the more trustworthy signals in the dataset precisely because the effect lands where domain knowledge says it should (food spending, not hobbies) — a genuine internal consistency check, not just a correlation found by chance.

Charts: `17_snap_effect_by_state.png`, `18_snap_effect_by_category.png`. Tables: `snap_effect_by_state.csv`, `snap_effect_by_category.csv`, `snap_effect_by_department.csv`, `snap_effect_by_store.csv`.

## 10. Price Findings (Phase 7)

**What was analyzed:** price distributions, price-change frequency, and — critically, given there is no promotion field — whether price changes are associated with sales changes, using both a within-item "relative price" view and a before/after comparison around detected price-change events.

**Why:** price is one of the few variables in this dataset that is legitimately known in advance for the forecast horizon (retailers set prices ahead of time), making it a potentially valuable predictive input if the relationship is well understood.

**What was found:**

- Prices range from $0.01 to $107.32 (mean $4.41); HOBBIES has the widest price spread (std $4.82), FOODS the narrowest (std $2.13, mean $3.25).
- Price changes are infrequent: 78,057 change events across the whole dataset, a mean of 2.56 changes per series over ~5 years, affecting a **0.16% daily change rate** among priced days. 57.3% of changes are increases, 42.7% decreases; the median absolute change is 9.4%.
- **Within-item relative price shows a mostly sensible pattern:** mean sales are 2.34 units when price is under 85% of the item's own average, falling to 1.29 units at 1.00-1.05x — consistent with basic demand theory. However, the pattern is **not perfectly monotonic**: sales tick back up to 1.56 units at 1.05-1.15x before falling again above 1.15x. This nuance is reported as-is rather than smoothed into a clean story.
- **A more surprising finding: sales rose after BOTH price increases and price decreases** when comparing a 7-day window before vs. after each change (price decreases: mean sales +48.5%; price increases: mean sales +71.0%). But the **median** change in both directions was **0%** — meaning the mean is driven by a subset of series where sales and price moved together, most series showed no material before/after difference, and price changes may often coincide with *other* demand-moving events (season, a new listing ramping up, etc.) rather than causing the lift themselves.

**This is explicitly a hypothesis, not a confirmed effect:** the dataset has no promotion field, so we cannot distinguish "the price change caused the demand change" from "both were driven by something else happening at the same time." A price-change feature is a plausible **demand-shock signal candidate**, not a validated causal driver.

Charts: `19_price_distribution_by_category.png`, `20_sales_by_relative_price_bin.png`, `21_relative_price_by_category.png`, `22_sales_before_after_price_change.png`. Tables: `price_distribution_by_category.csv`, `sales_by_relative_price_bin.csv`, `sales_by_relative_price_bin_and_category.csv`, `price_change_event_sales_before_after.csv`.

## 11. Product / Store Heterogeneity (Phase 8)

**What was analyzed:** whether series behave uniformly enough for one modeling strategy to suit all of them — combining the activity-class segmentation (Section 6) with sales-volume tiers, a per-series seasonality index, and a sampled per-series look at event and price sensitivity.

**Why:** if behavior is genuinely heterogeneous, a single global model/loss function is unlikely to serve every segment equally well — worth knowing before the team commits to an approach.

**What was found:**

- Crossing activity class with volume tier (tertiles of mean sales) shows real segments, not a smooth continuum — e.g., High-activity series skew toward higher volume, Extremely-sparse series skew low-volume, but overlap exists in both directions (full cross-tab: `EDA/tables/segment_activity_x_volume.csv`).
- Per-series **seasonality strength varies widely**: the coefficient of variation of monthly means ranges from 0.136 (p10, nearly flat) to 0.557 (p90, strongly seasonal) — a >4x spread. HOUSEHOLD and HOBBIES trend slightly more seasonal on average than FOODS (`EDA/tables/series_seasonality_index.csv`).
- In a stratified sample of 501 series, **event sensitivity was mixed but skewed negative**: 63.3% of sampled series showed *lower* sales on event days than non-event days (median -0.02 units), consistent with the Section 8 finding that many major named events (Christmas, Thanksgiving) suppress sales more than positive events (LaborDay) lift them.
- In the same sample (358 series had enough price variation to compute a correlation), **65.1% showed a negative price-sales correlation and 34.9% a positive one** — a majority-negative but far from unanimous pattern, and weakest in FOODS (mean r = -0.068) vs. near-zero in HOBBIES/HOUSEHOLD.

**Conclusion supported by this evidence:** behavior is genuinely heterogeneous across category, department, store, activity level, seasonality, and (to a lesser, noisier extent) price/event sensitivity. **A single one-size-fits-all forecasting approach is unlikely to perform equally well across all of these segments** — this is a data-supported statement, not an assumption, though the team should weigh it against added complexity.

Charts: `23_segment_activity_x_volume.png`, `26_sampled_price_sales_correlation_distribution.png`. Tables: `segment_activity_x_volume.csv`, `series_segment_assignment.csv`, `series_seasonality_index.csv`, `sampled_series_event_price_sensitivity.csv`.

## 12. Important Relationships (Phase 9)

**What was analyzed:** how strongly today's sales relate to (a) past sales at various lags, (b) recent rolling averages, (c) price, SNAP, and event status, and (d) how long a series has gone without a sale. All computed on the full 59.18M-row table.

**Why:** this directly informs which candidate features are worth prioritizing before any modeling begins.

**What was found (Pearson correlation with same-day sales, unless noted):**

| Relationship | Correlation / effect |
|---|---|
| lag_1 (yesterday's sales) | r = 0.768 |
| lag_7 | r = 0.720 |
| lag_14 | r = 0.689 |
| lag_28 | r = 0.672 |
| rolling_mean_7 (prior 7 days) | **r = 0.820 — the strongest relationship found in this entire EDA** |
| rolling_mean_28 (prior 28 days) | r = 0.807 |
| sell_price | r = -0.151 (weak; raw price is confounded by cross-item scale — see Section 10 for the within-item view) |
| is_snap_day | r = 0.017 (weak point-correlation, but the group-mean lift in Section 9 is real — diluted here by the mostly-zero target) |
| is_event_day | r = -0.004 (near zero in aggregate — masks large offsetting effects, see Section 8) |

- **The single cleanest relationship in the whole EDA:** the probability of a sale today falls in a near-perfect staircase as the length of a series' current dry spell grows — **65.2%** chance of a sale if it sold yesterday, dropping to **38.3%** (1-3 days dry), **22.2%** (4-7), **12.7%** (8-14), **6.2%** (15-28), and just **0.6%** after 29+ consecutive zero days (`EDA/charts/25_days_since_last_sale_vs_today.png`).

**Why it matters:** recent history (both raw lags and, even more strongly, rolling averages) and the current dry-spell length are by far the most informative signals found. Calendar-based signals (price, SNAP, events) are real but individually much weaker in raw correlation terms — their value lies more in specific, well-targeted forms (named events, relative price, state-matched SNAP) than as blunt aggregate flags.

Charts: `24_lag_correlations.png`, `25_days_since_last_sale_vs_today.png`. Table: `days_since_last_sale_vs_today.csv`.

## 13. Candidate Feature Engineering (Phase 10)

27 candidate features were evaluated, each grounded in a specific finding above. Full detail — including data source and priority for every feature — is in `EDA/tables/feature_candidates.csv`; the table below summarizes by priority.

**High priority** (strong, direct EDA evidence): `day_of_week`, `lag_1`* , `lag_7`*, `lag_28`, `rolling_mean_7`*, `rolling_mean_28`*, `days_since_last_sale`, `zero_streak_length`, `recent_nonzero_rate`, `event_name`, `snap_indicator` (state-matched), `days_since_first_listing`, `category`/`department`/`store`/`state` hierarchy fields. (*see Section 14 for the origin-relative caveat on lags and rolling stats.)

**Medium priority** (real but modest effect, or needs careful construction): `is_weekend`, `month`, `year`/time trend (see leakage note below), `rolling_std_7`/`rolling_std_28`, `activity_class`, `current_price`, `price_relative_to_recent_average`, `seasonality_cv`.

**Low priority / not yet recommended**: `week_of_year` (real effect but high cardinality relative to data), `price_change`/`price_change_pct` (correlational only, no promotion field to confirm causality — flagged as hypothesis), `is_event_day` (binary flag; weak standalone, better used via `event_name`), `event_type_1` (modest effect), `event_proximity` (untested — no EDA evidence yet, would need its own analysis before adoption).

**These are candidates only.** The team should choose which to actually build based on the modeling strategy, not this list alone.

## 14. Feature Leakage Considerations (Phase 11)

This is the most important section for correctness. For **every** candidate feature, the question asked was: *would we actually know this value when standing at the forecast origin, trying to predict the next 28 days?*

**The central subtlety: this is a fixed-origin, 28-day-ahead forecast, not a sequence of one-step-ahead predictions (unless the team chooses a recursive strategy).** That changes which lag/rolling features are safe:

- **`lag_1`, `lag_7`, `lag_14` are only safe if computed once relative to the forecast origin and held constant across all 28 target days**, or used inside a recursive (day-by-day) forecasting loop. Recomputing them "relative to each target day" — i.e., using each target day's own recent history — would require already knowing near-future sales for that target day. **That is leakage.**
- **`lag_28` (origin - 28 days) is the only single lag that is naturally safe for every one of the 28 horizon days** from a fixed origin, because it never reaches past the origin date.
- **`rolling_mean_7`/`rolling_mean_28` and `rolling_std_7`/`rolling_std_28` must be computed using only days up to and including the origin**, then held constant across the whole horizon — never recomputed with dates inside the forecast window.
- **`days_since_last_sale`, `zero_streak_length`, `recent_nonzero_rate`, `activity_class`, and `seasonality_cv`** are all safe **only if** computed strictly from history available at the origin. Computing any of these using the *full* series history (including the period being forecast) would leak the target period's own behavior into its own feature — a subtle but serious risk since these are the strongest features found (Section 12).
- **`price_relative_to_recent_average`** has the same risk: the "item's own average price" must come from an expanding or trailing window ending at the origin, not the full series.
- **`current_price` for the forecast horizon is likely genuinely available** (Walmart prices are set in advance, and `sell_prices.csv` covers weeks beyond the processed table's last day per `PROCESSING_REPORT.md`) — but `sales_long_full.parquet` as built does **not** yet include those future-horizon price rows; a separate join would be needed.
- **`event_name`, `event_type_1`, SNAP indicators, and all calendar-derived fields (day_of_week, month, week_of_year) are safe without qualification** — the calendar is fully known in advance for the entire horizon.
- **Raw categorical `year` is a leakage-adjacent risk of a different kind**: any forecast horizon will contain year values (2016 partial, or beyond) with limited or no training examples, so a plain categorical encoding will not generalize. A continuous time index is a safer alternative, but see Section 7's composition-effect caveat before treating it as pure "trend."
- **`price_change`/`price_change_pct`** are flagged Low priority in Section 13 primarily *because* of the causal-ambiguity finding in Section 10 (Section 11 leakage risk is secondary here — the bigger problem is that the correlational finding itself is confounded, not just that it's hard to compute safely).

No feature in the candidate table is recommended for use without the origin-relative caveat noted in its row of `feature_candidates.csv`.

## 15. Potential Novelty Directions

**These are hypotheses for team discussion, not a decision.** Each is directly motivated by a specific EDA finding above.

1. **Pre-listing-aware zero handling.** The near-perfect alignment between leading zero blocks and leading no-price periods (Section 6) suggests a structural way to separate "not yet listed" zeros from "listed but no demand" zeros — something a naive model cannot distinguish on its own. A novelty direction could explore whether explicitly modeling (or masking) this pre-listing period improves forecasts for newer items.
2. **Intermittent-demand-aware modeling using dry-spell state.** The days-since-last-sale relationship (Section 12) is the strongest single signal found — far stronger than any calendar feature. A novelty direction could center the whole modeling approach around this state (e.g., a two-stage "will it sell / how much if it sells" structure, common in intermittent-demand forecasting) rather than treating sales as an ordinary continuous target.
3. **Named-event-aware forecasting, not generic holiday flags.** Section 8 showed that a blunt "is_event" flag actively hides information (Christmas and LaborDay point in opposite directions by over 100 percentage points). A novelty direction could build a per-event effect model rather than a single event coefficient.
4. **Segment-specific or hierarchical modeling exploiting confirmed heterogeneity.** Section 11 found real, data-supported behavioral differences across activity class, category, seasonality, and (more weakly) price/event sensitivity. A novelty direction could pool statistical strength within segments (e.g., hierarchical/multi-level models, or per-segment feature sets) rather than a single global model.
5. **Price-change as a weak "demand-shock" signal, used cautiously.** Section 10's finding — sales rise after both price increases and decreases, with a median effect of zero — suggests price changes may mark "something is happening" periods rather than cleanly causing demand shifts. A novelty direction could treat detected price changes as an uncertainty/attention signal (e.g., wider prediction intervals nearby) rather than a direct causal feature.

## 16. Key Findings for Team Discussion

| # | Finding | Evidence | Why it matters | Modeling implication |
|---|---|---|---|---|
| 1 | 68.0% of all rows are zero sales; the target is heavily zero-inflated and right-skewed (skewness 17.1) | Phase 1/2, `phase1_sanity_check.json` | Standard regression assumptions (normal, homoscedastic errors) don't fit | Consider count/zero-inflated models, or a two-stage occurrence+magnitude approach |
| 2 | Top 10% of series drive 54.4% of total sales; bottom 10% drive 0.52% | Phase 2, `series_level_sales_stats.csv` | Aggregate accuracy metrics can hide poor performance on the long tail | Consider volume-aware evaluation, not just overall RMSE/WAPE |
| 3 | FOODS is 68.6% of total sales; FOODS_3 alone is 49.2% | Phase 2, `sales_by_department.csv` | A global model will be dominated by one department unless corrected for | Consider category-aware loss weighting or per-category evaluation |
| 4 | Per-series zero% ranges from single digits to >95% — no one-size-fits-all "typical" series | Phase 3, `series_zero_pct.csv` | Confirms genuine heterogeneity, not just category-level averages | Segment-aware or hierarchical strategy worth considering |
| 5 | Leading zero blocks align with leading no-price periods (median gap 3 days, 99.48% within a week) | Phase 3, `series_leading_zero_and_price_gap.csv` | Strong, specific evidence these are "not-yet-listed" periods, not ordinary demand gaps | Consider excluding/flagging pre-listing periods rather than training on them as normal zeros |
| 6 | Zero-run lengths are themselves heavy-tailed: median 2 days, max 1,845 days | Phase 3, `zero_streak_length_distribution.csv` | A single "average zero rate" hides very different dry-spell dynamics | Streak-based features likely more informative than a flat zero-rate feature |
| 7 | Weekend sales are 31.1% higher than weekday sales, consistently across categories | Phase 4 | One of the strongest, cleanest calendar effects found | High-priority, low-risk feature |
| 8 | Yearly sales growth (2011-2015) is confounded with assortment growth (16,762 → 30,474 active series) | Phase 4, composition check | A raw year/trend feature risks learning catalog growth, not real demand growth | Any trend feature needs the composition effect controlled for |
| 9 | 2016 is a partial year (ends 2016-05-22) | Phase 4/1 | Naive full-year comparisons involving 2016 will be misleading | Always filter or annualize 2016 comparisons |
| 10 | Individual named events have very large, opposite-direction effects (Christmas -99.95%, LaborDay +27.5%) hidden by a small, misleading aggregate event effect (-4.6% overall) | Phase 5 | A blunt event flag destroys almost all the useful signal | Use event identity (`event_name`), not just an event flag |
| 11 | SNAP effect (+12.7% overall) concentrates specifically in FOODS (+17.3%, and +32.3% in FOODS_2) | Phase 6 | Effect lands exactly where domain knowledge predicts — a genuine internal consistency check | High-confidence, low-risk feature |
| 12 | Within-item relative price shows the expected direction for the most part, but is not perfectly monotonic | Phase 7 | Price-demand relationship is real but noisier/more complex than a simple linear story | Worth using, but validate carefully rather than assuming linearity |
| 13 | Sales rise after BOTH price increases and decreases (mean +71%/+48%) while the median change is 0% | Phase 7 | Simple before/after price-change comparisons are likely confounded by co-occurring events, not a clean causal signal | Treat price-change as a hypothesis/shock signal, not a validated causal feature |
| 14 | rolling_mean_7 (r=0.820) and rolling_mean_28 (r=0.807) are the strongest predictors found — stronger than any single lag | Phase 9 | Recent average level matters more than any single day's value | Prioritize rolling-window features over single-day lags where feasible |
| 15 | P(sale today) falls from 65.2% to 0.6% as dry-spell length grows from 0 to 29+ days — a near-perfect staircase | Phase 9 | The single cleanest relationship in the entire EDA | `days_since_last_sale` should be a top-priority feature |
| 16 | 51.4% of series have CV > 2 — more than half the catalog is highly volatile relative to its own mean | Phase 2 | Point forecasts alone may be insufficient for much of the catalog | Consider quantile/interval forecasting, not just point estimates |
| 17 | This is a fixed-origin, 28-day-ahead task, so most lag/rolling features are only safe relative to the origin, not per-target-day | Phase 11 | The single biggest correctness risk identified in this EDA | Feature-engineering must define every lag/rolling feature relative to the forecast origin, or use a recursive strategy deliberately |
| 18 | ~20.78% of rows have no matched price, mostly reflecting pre-listing periods (consistent with Finding 5) | Processing report + Phase 3 | Missing price is informative, not random | Do not blindly impute price; consider using its missingness as a feature |
| 19 | Sampled per-series price/event sensitivity is directionally consistent with aggregate findings but noisy and far from unanimous (65%/35% split on price direction) | Phase 8 | Category/store-level aggregate effects don't necessarily hold uniformly at the individual-series level | Be cautious about assuming aggregate relationships apply to every series equally |
| 20 | Raw categorical `year` will contain unseen future values; a plain category encoding will not generalize to the forecast horizon | Phase 11 | Straightforward but easy-to-miss correctness issue | Use a continuous time index if a trend feature is wanted, with Finding 8's caveat in mind |

## 17. Limitations

- This EDA characterizes **historical, already-observed data only** (`d_1`-`d_1941`, 2011-01-29 to 2016-05-22). It says nothing directly about how these patterns will hold in the 28-day period to be forecast.
- The event-lift, price-change, and per-series sensitivity analyses are all **correlational**. None of them establish causation, and none should be read as such.
- The per-series event/price sensitivity results (Section 11) are based on a **sample of 501 of 30,490 series** — informative about the overall shape of the distribution, not a claim about any specific series.
- The "pre-listing" interpretation of leading zero blocks (Section 6) is a well-evidenced hypothesis, not a certainty — the dataset has no direct "listed/not listed" field to confirm it definitively.
- `event_proximity` and several other Low-priority candidate features in Section 13 are genuinely untested — their absence of evidence here reflects lack of analysis, not evidence of no effect.
- 2016 data is partial (through May 22 only); any statistic that aggregates by full calendar year should account for this.

## 18. What Was NOT Done

Per the scope of this stage, none of the following were performed:

- No forecasting model was trained, and no algorithm was selected.
- No hyperparameter tuning was performed.
- No final 28-day forecast was produced.
- No final ML training dataset (with engineered features) was created — `sales_long_full.parquet` remains exactly as built in the prior processing stage.
- No feature listed in Section 13 was actually computed and added to any dataset — this report only evaluates candidates.
- No project novelty was chosen — Section 15 lists directions for team discussion only.
- No zero values were removed, replaced, or reinterpreted; no sales spikes were removed as outliers; no stockout or promotion labels were invented.
- `raw_dataset/` and `processed_dataset/sales_long_full.parquet` were not modified — see Final Validation below.

## 19. Recommended Next Step

Discuss the findings and candidate features in this report as a team, then jointly decide: (a) the forecasting strategy (e.g., global vs. segment-specific, recursive vs. direct multi-horizon — which directly determines which lag/rolling features in Section 14 are usable), (b) which candidate features from Section 13 to actually build, respecting the leakage constraints in Section 14, and (c) which of the five novelty directions in Section 15 (if any) the team wants to pursue. Only after that should feature engineering and the final ML training dataset be built.

---

## Final Validation

Performed after all analysis scripts completed:

| Check | Result |
|---|---|
| `raw_dataset/` file checksums vs. `processed_dataset/PROCESSING_REPORT.md` record | Unchanged (see `EDA/statistics/audit_metadata.json`) |
| `sales_long_full.parquet` checksum before vs. after this EDA stage | Unchanged |
| Row/column shape of `sales_long_full.parquet` | Unchanged: 59,181,090 x 22 |
| Zero-sales count recomputed fresh in Phase 1 vs. `PROCESSING_REPORT.md` | Identical: 40,241,819 |
| Sales sum recomputed fresh in Phase 1 vs. `PROCESSING_REPORT.md` | Identical: 66,927,173 |
| Every chart traces to a script that wrote its numbers to `EDA/statistics/*.json` or `EDA/tables/*.csv` first | Confirmed by construction — see `EDA_METHODOLOGY.md` |
| `feature_candidates.csv` leakage assessments reviewed against forecast-origin availability | Confirmed - see Section 14 |
| No future sales information used in any statistic | Confirmed - all analysis restricted to `d_1`-`d_1941`; no `sample_submission.csv` or post-horizon data referenced |

---

*Generated as part of the EDA / feature-design investigation stage. No model training, algorithm selection, hyperparameter tuning, final forecast, final ML dataset creation, or novelty decision was made in this stage.*
