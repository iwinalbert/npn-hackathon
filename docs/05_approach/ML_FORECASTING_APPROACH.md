# Listing-Aware Hurdle Forecasting with Recency State

### ML Modeling Strategy — Planning & Design Document

*Problem Statement 11 · M5 Retail Demand Forecasting · NPN AIA Hackathon — St. Joseph's*

> **PLANNING / DESIGN STAGE ONLY  —  NO MODEL HAS BEEN TRAINED  —  NO FINAL FEATURE DATASET HAS BEEN BUILT**

*Document generated: Friday, August 14, 2026*

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Context](#project-context)
3. [Confirmed EDA Findings](#confirmed-eda-findings)
4. [Proposed Modeling Approach](#proposed-modeling-approach)
5. [Stage 1 — Sale Occurrence Model](#stage-1-sale-occurrence-model)
6. [Stage 2 — Demand Magnitude Model](#stage-2-demand-magnitude-model)
7. [Proposed Model Family](#proposed-model-family)
8. [Baseline Model](#baseline-model)
9. [Validation and Backtesting](#validation-and-backtesting)
10. [Data Leakage](#data-leakage)
11. [Feature Engineering Plan](#feature-engineering-plan)
12. [Zero Handling](#zero-handling)
13. [Promotion and Event Handling](#promotion-and-event-handling)
14. [Novelty Statement](#novelty-statement)
15. [Experiment Plan](#experiment-plan)
16. [Metrics](#metrics)
17. [Final 28-Day Forecast Process](#final-28-day-forecast-process)
18. [Project Pipeline](#project-pipeline)
19. [Team Decisions Required](#team-decisions-required)
20. [Glossary](#glossary)
21. [Appendix A — Source Documents Examined](#appendix-a-source-documents-examined)
22. [Appendix B — Confirmed Findings vs. Proposed Hypotheses](#appendix-b-confirmed-findings-vs-proposed-hypotheses)

---

## Executive Summary

This document is a planning and design document. It exists so the team can discuss and agree on a modeling strategy for Problem Statement 11 before any implementation begins. No model has been trained, no final feature dataset has been created, and no prediction has been submitted. Nothing in raw_dataset/ or processed_dataset/ was changed while preparing this document.

We need to forecast daily unit sales for 30,490 store-item series, 28 days into the future (2016-05-23 to 2016-06-19), using about 5.3 years of daily history (2011-01-29 to 2016-05-22). Our own exploratory data analysis (EDA) found that the data is dominated by zero-sales observations (68.0% of all rows), is highly skewed, and behaves differently across products, stores, and time. These findings, not general assumptions about retail data, are what shape the plan below.

> **PROPOSED / HYPOTHESIS — NOT YET PROVEN:** Our current proposed direction is called “Listing-Aware Hurdle Forecasting with Recency State.” This is a proposed novelty, not a proven final solution. It must be tested against a simpler baseline before the team adopts it.

In short, the plan is: build a simple baseline first, build the proposed two-stage model second, compare them honestly using historical data we already have real answers for, and only keep the added complexity if it actually earns its place. Every technical term used in this document is explained in plain English the first time it appears, and a full glossary is provided in the appendix.

**What we know (confirmed by EDA):**

- 68.0% of all daily sales records are exactly zero (EDA Phase 1/3)
- rolling_mean_7 and days_since_last_sale are our strongest known predictive signals (EDA Phase 9)
- Calendar, events, SNAP, and price are already available for the full 28-day forecast window
- There is no promotion flag and no stockout flag anywhere in the dataset

**What we're assuming (not yet proven — needs testing):**

- That a two-stage (hurdle) model will outperform a single model — not yet tested
- That “pre-listing” zeros can be reliably separated from ordinary intermittent zeros
- That segment-specific or listing-aware features will improve accuracy enough to justify their complexity
- The exact scoring metric for this hackathon (still unconfirmed)

## Project Context

Problem Statement 11 asks us to forecast item-level demand across stores and product hierarchies, balancing stockouts against overstocking, using real Walmart-style data: 3,049 products across 10 stores in 3 US states (California, Texas, Wisconsin), with intermittent (sparse, zero-inflated) demand patterns. The data source is the M5 Forecasting — Accuracy competition on Kaggle.

### Files we start from

| File | Rows | What it is |
|---|---|---|
| calendar.csv | 1,969 | One row per date; weekday, month, year, events, SNAP flags |
| sales_train_validation.csv | 30,490 | Daily sales through 2016-04-24 (1,913 days) — a strict subset of the evaluation file |
| sales_train_evaluation.csv | 30,490 | Daily sales through 2016-05-22 (1,941 days) — our primary historical source |
| sell_prices.csv | 6,841,121 | Weekly selling price per store-item, including the 28 future days |
| sample_submission.csv | 60,980 | Required output shape: id, F1…F28 |

### Where the project currently stands

The team has already completed two stages before this one: a data processing stage that joined the raw files into a single long-format table, and an EDA stage that characterized the resulting data. This document is the third stage: turning those findings into an agreed modeling plan, before any feature engineering or training happens.

```
raw_dataset/  (original M5 files, untouched)
    |
    v
processed_dataset/sales_long_full.parquet  (59,181,090 rows × 22 columns, joined & validated)
    |
    v
analysis_output/  +  EDA_REPORT  (patterns and candidate features identified)
    |
    v
THIS DOCUMENT: ml_strategy/  (modeling plan — for team discussion and approval)
```

> **Why this matters:** Every number and finding used from this point forward is taken from the team's own PROCESSING_REPORT and EDA_REPORT — not invented, and not assumed from general knowledge of the public M5 competition.

### The forecasting timeline

| Period | Dates | What it means |
|---|---|---|
| Historical (known) | 2011-01-29 → 2016-05-22 | 1,941 days of real, observed sales |
| Forecast horizon (unknown) | 2016-05-23 → 2016-06-19 | 28 days — the actual target we must predict |

## Confirmed EDA Findings

This section restates, in plain English, the findings our own EDA already established. Every item below is confirmed by a statistic, table, or chart already produced in EDA_REPORT — nothing here is a new claim. Later sections build on these findings; wherever this document goes beyond them, it is explicitly marked as a hypothesis instead.

### 1. Most day-item-store combinations have zero sales

> **Term explained — Zero sales:** no units were recorded as sold for that particular store, item, and day. This does not necessarily mean nothing happened — it just means the recorded sales count for that row is 0.

68.0% of all 59,181,090 daily observations are exactly zero (mean sales = 1.131, median = 0.0, skewness = 17.1 across all rows and 11.9 even after dropping the zeros). This is the single most defining property of the dataset. We should not assume every zero means the same thing — different types of zero periods may exist, and Section 12 (Zero Handling) investigates this directly.

### 2. Leading zero periods line up with missing price history

For 99.48% of the 30,490 series, the length of the leading (starting) run of zero sales is within 7 days of the length of the leading period with no recorded price (median difference: just 3 days). This is consistent with the hypothesis that some early zeros reflect a product not yet being active in a store's assortment, rather than genuine no-demand.

> **CONFIRMED BY EDA:** Leading zero-sales gaps and leading no-price gaps are closely aligned for the large majority of series.

> **PROPOSED / HYPOTHESIS — NOT YET PROVEN:** That this alignment means the product was “not yet listed.” This is a reasonable, well-supported explanation — but it is an interpretation of the pattern, not a labeled fact in the data. The dataset has no field that directly states when a product was listed.

### 3. Sales are highly intermittent

> **Term explained — Intermittent demand:** a pattern where a product does not sell every day. It may sell several units on one day, then have zero sales for several days, then sell again.

Zero-percentage varies by category (HOBBIES 77.1% vs. FOODS 61.8%) and by store (CA_3 59.4% vs. CA_4 72.0%). Looking at each series individually, the zero-percentage ranges widely: 24.7% at the 5th percentile up to 95.0% at the 95th percentile, with a median of 73.3%. Splitting series into activity classes based on this distribution gives 7,629 High-activity, 15,267 Regular/intermittent, 6,103 Sparse, and 1,491 Extremely sparse series.

### 4. The longer it's been since the last sale, the less likely a sale is today

> **Term explained — days_since_last_sale:** a feature that tells the model how many days have passed since this exact store-item combination last recorded a sale.

Probability of a sale today falls in a near-perfect staircase as the dry spell lengthens: 65.2% if it sold yesterday, 38.3% after a 1–3 day gap, 22.2% after 4–7 days, 12.7% after 8–14 days, 6.2% after 15–28 days, and just 0.6% after 29+ consecutive zero days. Zero-run lengths across the dataset have a median of 2 days but a mean of 6.08 days (long tail up to 1,845 days for some series).

### 5. Recent sales are highly predictive of today's sales

> **Term explained — Rolling mean:** the average sales over a recent window of time, such as the previous 7 or 28 days.

rolling_mean_7 has the strongest correlation with same-day sales found anywhere in the EDA (r = 0.820), followed closely by rolling_mean_28 (r = 0.807). By comparison, single-day lag features are weaker: lag_1 (r = 0.768), lag_7 (r = 0.720), lag_14 (r = 0.689), lag_28 (r = 0.672).

> **Term explained — Lag feature:** a previous day's or week's sales value used as an input. lag_1 = yesterday's sales, lag_7 = sales exactly 7 days ago, lag_28 = sales exactly 28 days ago.

### 6. Weekends sell more than weekdays

Mean sales on Saturday/Sunday (1.361) are 31.1% higher than the weekday-only mean (1.038), and this pattern holds within every product category. Monthly effects are real but much smaller — the highest month (#8) is only 8.99% above the lowest (#12). The apparent year-over-year growth in total sales (2011–2015) is partly a composition effect: the number of active series nearly doubled over that window (16,762 → 30,474), so some of the “growth” is more products being listed, not existing products selling more.

### 7. SNAP (a food-assistance benefit) is linked to higher sales

> **Term explained — SNAP:** the Supplemental Nutrition Assistance Program, a US government food-assistance benefit. In this dataset, a snap_CA / snap_TX / snap_WI flag indicates whether that benefit was usable in that state on that day.

SNAP days show 12.7% higher mean sales overall than non-SNAP days. The effect concentrates in food: FOODS +17.3% (FOODS_2 alone +32.3%, the largest of any department) vs. HOUSEHOLD +3.5% and HOBBIES +2.5%. The direction holds in all three states, though the size varies: WI +21.8%, TX +11.5%, CA +8.0%. This is one of the more trustworthy signals in the whole EDA, precisely because the effect lands exactly where basic domain knowledge would predict (food spending) — a genuine internal consistency check.

### 8. Named calendar events affect sales very differently from each other

In aggregate, event days look almost unremarkable (−4.6% vs. non-event days) — but that average hides large, opposing effects underneath. Christmas is −99.95% vs. local baseline (stores are essentially closed); Thanksgiving −29.9%; New Year −20.4%; Mother's Day −12.5%. Meanwhile Labor Day is +27.5%, Columbus Day +9.6%, Veterans Day +8.8%. A single generic “is this an event day” flag would badly underuse this signal — the identity of the specific event carries the real information.

> **Why this matters:** This is exactly why we should not treat all events (or, by extension, all price changes) as one generic “promotion” signal — the effects point in opposite directions and are event-specific.

### 9. Price changes are infrequent, and their effect is not fully understood yet

Price changes are rare: 78,057 change events total, averaging 2.56 changes per series over ~5 years (a 0.159% daily change rate). Lower relative prices are mostly, but not perfectly, associated with higher sales — there's an unexplained uptick in the 1.05–1.15x relative-price band. More surprisingly, sales rose after both price increases (+71.0% mean) and price decreases (+48.5% mean), while the median change in both directions was 0%. This suggests price changes often coincide with some other demand-moving event, rather than directly causing the change themselves.

> **PROPOSED / HYPOTHESIS — NOT YET PROVEN:** That price changes can act as a weak, indirect signal for unusual demand. There is no promotion field in the dataset to confirm this causally, so price should be used cautiously, not treated as a stand-in for “this item was on sale.”

Sampled evidence (n = 501 of 30,490 series, stratified by category and volume) found roughly 65% of series show a negative price-sales correlation and 35% show positive — majority-negative, but far from unanimous.

### 10. The dataset is genuinely heterogeneous

Different products, categories, stores, and activity levels behave differently, and this shows up at every level examined: per-series seasonality strength (coefficient of variation of monthly means) spans a more than 4x range (0.136 at the 10th percentile to 0.557 at the 90th); in the sampled event-sensitivity check, 63.3% of series actually showed lower sales on event days, not higher. Behavior is genuinely heterogeneous across category, department, store, activity level, seasonality, and (more weakly) price/event sensitivity — a data-supported conclusion, not an assumption.

> **Why this matters:** A single one-size-fits-all forecasting approach is unlikely to perform equally well across all these different segments. This is the direct motivation for considering segment-aware or listing-aware modeling in the sections that follow.

## Proposed Modeling Approach

> **PROPOSED / HYPOTHESIS — NOT YET PROVEN:** Everything in this section is a PROPOSED novelty — not a proven final solution. It is our current best hypothesis, built directly from the EDA findings above, and it must be tested against a baseline before the team relies on it.

Our current proposed direction is called “Listing-Aware Hurdle Forecasting with Recency State.” The name sounds technical, but each part describes a specific, simple idea:

### “Listing-aware”

The model considers whether a product appears to have become active / listed in a store's assortment, rather than treating every early zero-sales period identically. This responds directly to EDA Finding 2 (leading zero periods aligning with missing price history).

### “Hurdle model”

> **Term explained — Hurdle model:** a two-stage forecasting approach. Instead of asking one model to answer one question (“how many units will be sold?”), we ask two separate, simpler questions and then combine the answers.

Stage 1 asks: “will this product sell at all, on this day?” Stage 2 asks: “if it does sell, how many units?” The two answers are then multiplied together to produce the final forecast.

> **Formula:** `Final forecast  =  P(sale)  ×  E(units | sale)`
>
> **Worked example:** If the model estimates a 75% chance of a sale (P(sale) = 0.75), and expects 8 units on days when a sale happens (E(units | sale) = 8), then: Final forecast = 0.75 × 8 = 6 units.

> **Why this matters:** With 68% of all observations at zero, predicting whether a sale happens at all and predicting how large that sale is are related but genuinely different problems. Forcing one model to answer both at once is exactly the kind of thing that can produce a muddled, overly-smoothed forecast on data shaped like ours.

### “Recency state”

The model is given explicit information about how recently this store-item last sold (days_since_last_sale, zero_streak_length), directly responding to EDA Finding 4, our single cleanest relationship in the whole investigation.

**What we know (confirmed by EDA):**

- 68% zero-inflation makes occurrence and magnitude genuinely different sub-problems (Finding 1)
- days_since_last_sale shows a near-perfect staircase relationship with P(sale) (Finding 4)
- rolling_mean_7 / rolling_mean_28 are the strongest predictors found in the EDA (Finding 5)

**What we're assuming (not yet proven — needs testing):**

- That splitting occurrence from magnitude will out-forecast a single combined model — to be tested (Experiment 3 vs 2)
- That a reliable “listing-aware” feature can be built from the price-history proxy — to be tested (Experiment 4)

## Stage 1 — Sale Occurrence Model

> **Term explained — Classification:** a model that predicts a category or outcome (for example, yes/no) rather than directly predicting a numeric quantity.

Stage 1 is a binary classification model. Its target is simple: for a given store-item-day, did sales > 0 or sales = 0?

### Candidate features for Stage 1

These are candidates to be tested during feature engineering, not a final, approved list.

| Group | Candidate features |
|---|---|
| Recency | days_since_last_sale, zero_streak_length, recent sales, rolling_mean_7, rolling_mean_28 |
| Listing / activity | whether the product appears active/listed; days_since_first_listing (only if we can define this reliably — see Section 12) |
| Calendar | day of week, month, year, weekend indicator |
| Event | event name, event type |
| SNAP | state-matched SNAP indicator |
| Hierarchy | store, state, category, department, item |

> **Why this matters:** Not all of these will necessarily end up in the final model — each one must earn its place through validation (Section 15, Experiment Plan), not be included by default.

## Stage 2 — Demand Magnitude Model

> **Term explained — Regression:** a model that predicts a numeric quantity directly (as opposed to classification, which predicts a category).

Stage 2 answers: “how many units are expected to be sold, assuming the product sells at all?” It is trained primarily on the subset of observations where sales > 0.

### Candidate features for Stage 2

- Recent sales, rolling averages, lag features (see definitions below)
- Price, and price relative to the item's own recent average
- Calendar and event information
- SNAP indicator
- Store / category / item hierarchy
- Activity / recency information (same recency features as Stage 1)

> **Term explained — Lag feature:** a previous day's or week's sales value. lag_1 = yesterday's sales, lag_7 = sales 7 days ago, lag_28 = sales 28 days ago.

> **Term explained — Rolling feature:** a feature that summarizes recent history, such as an average. rolling_mean_7 = the average of sales over the previous 7 days.

## Proposed Model Family

The current practical candidate for both stages is LightGBM.

> **Term explained — LightGBM:** a fast machine-learning algorithm based on decision trees. It can learn non-linear relationships between many features and is practical to run on datasets as large as ours (59+ million rows).

| Stage | Candidate model | Notes |
|---|---|---|
| Stage 1 (occurrence) | LightGBM classifier | Standard binary classification setup |
| Stage 2 (magnitude) | LightGBM regression | Using an appropriate loss such as Tweedie, if it proves justified |

> **Term explained — Loss function:** the mathematical method used to measure how wrong a model's predictions are during training, which the model then tries to minimize.

> **Term explained — Tweedie:** a loss / distribution family that can be useful for non-negative, skewed data with many zeros — which matches our sales data reasonably well on paper. We still need to test whether it performs well on our dataset specifically.

> **PROPOSED / HYPOTHESIS — NOT YET PROVEN:** LightGBM and Tweedie loss are our current practical candidates, not a confirmed final choice. If experiments show a different approach works better, we will switch.

## Baseline Model

> **Term explained — Baseline:** a simpler model or forecasting method used purely as a reference point, so we can tell whether a more complex approach is actually worth its added complexity.

Before we can claim our proposed novelty works, we must first build a baseline and beat it honestly.

- Primary baseline: a single, standard LightGBM model trained on ordinary sales, calendar, and price features (no hurdle split, no listing-awareness).
- Secondary baseline (if practical): a simple seasonal-naive forecast — e.g., predicting each day using the same weekday's sales from a recent prior period.

> **TEAM DECISION NEEDED:** If the proposed hurdle / listing-aware model does not outperform the baseline on our validation setup, the added complexity is not justified, and we should reconsider or simplify it.

## Validation and Backtesting

We cannot simply train on all historical data and test on the true competition period, because we don't know the real future sales yet. Instead, we simulate the real competition situation using historical data we already have.

> **Term explained — Backtesting:** pretending that an earlier point in history is “today,” making a forecast from that point forward, and then checking the forecast against sales that actually happened afterward (which we already have on record).

### How a single backtest window works

| Split | Contents |
|---|---|
| TRAIN | All information before a chosen historical cutoff date |
| VALIDATION | The 28 known days immediately after that cutoff |
| COMPARE | Predicted sales for those 28 days vs. the actual sales we already have on record |

> **Why this matters:** Our dataset conveniently gives us one such window for free: days 1914–1941 are real, observed sales that were only added in the evaluation file. Cutting training off at day 1913 and validating against days 1914–1941 recreates the exact shape of the real competition task, using data whose true answers we already know.

### Why use multiple historical windows, not just one

A model might perform well on one particular 28-day window purely by luck — for example, if that window happened to avoid any unusual holidays. Testing across multiple different historical 28-day windows checks whether an approach works consistently, not just in one favorable case.

> **TEAM DECISION NEEDED:** Exactly how many historical windows to backtest against, and which ones (e.g., windows that include a major event like Christmas vs. windows that don't), still needs to be agreed.

## Data Leakage

> **PROPOSED / HYPOTHESIS — NOT YET PROVEN:** This is one of the most important sections in this document. Getting this wrong would make our validation results look better than they actually are.

> **Term explained — Data leakage:** when a model accidentally receives information during training or feature-building that would not actually be available at the real moment a forecast is made.

> **Example:** If we are predicting sales for May 25th, we cannot calculate a feature using the actual sales value from May 26th — because on the real forecast date, May 26th hasn't happened yet.

For a 28-day-ahead forecast, this is more subtle than it sounds, because lag and rolling features need to be built carefully relative to a single fixed starting point, not recomputed day-by-day using data that wouldn't really be available yet.

### Two possible strategies (not yet chosen)

| Strategy | How it works |
|---|---|
| 1. Recursive forecasting | Predict Day 1 of the horizon, then use that Day 1 prediction as an input to help predict Day 2, and so on through Day 28. |
| 2. Direct / fixed-origin forecasting | Use only information available at the forecast origin (the last known day) to construct predictions for all 28 horizon days at once, without ever using future actual sales. |

> **TEAM DECISION NEEDED:** Which of these two strategies we use is not yet decided. This choice directly determines which lag/rolling features are even usable — for example, lag_28 (28 days before the forecast origin) is the only single lag that is naturally safe across the entire 28-day horizon under a direct/fixed-origin approach without extra care.

> **Why this matters:** This decision should be made deliberately, with the team's agreement — not defaulted into accidentally by whichever features happen to be easiest to compute.

## Feature Engineering Plan

Feature engineering happens after the team approves the modeling strategy in this document — no features have been built yet. The groups below are candidates, organized by theme, grounded in the EDA findings from Section 3.

| Group | Candidate features |
|---|---|
| A. Historical demand | lag_1, lag_7, lag_14, lag_28 |
| B. Rolling demand | rolling_mean_7, rolling_mean_28, rolling_std |
| C. Recency | days_since_last_sale, zero_streak_length |
| D. Listing / activity | listing/activity indicator; days_since_first_listing (only if reliably defined) |
| E. Calendar | day of week, weekend indicator, month, year |
| F. Event | event name, event type |
| G. Price | current price, recent average price, price change, price relative to recent average |
| H. SNAP | state-specific SNAP indicator |
| I. Hierarchy | item, department, category, store, state |

> **TEAM DECISION NEEDED:** Not all of these features automatically belong in the final model. Each one will be tested through validation (Section 15), and only kept if it demonstrably helps.

## Zero Handling

> **PROPOSED / HYPOTHESIS — NOT YET PROVEN:** This section is extremely important, and easy to get wrong.

We are explicitly NOT going to replace zeros with averages, remove zeros, or smooth the sales data. The true target values must be preserved exactly as recorded — zeros are genuine, valid observations. What we are trying to do instead is understand the context and likely meaning behind different zero periods.

### Three possible types of zero, in plain English

1. Pre-listing / inactive period — the product was possibly not yet active in that store's assortment. (Hypothesis, supported by Finding 2.)
2. Active intermittent demand — the product is on sale and available, but simply had no customer purchase that day. (This is the normal, expected case for most of the dataset, per Finding 3.)
3. Possible stockout — a product may have been unavailable to buy despite real customer demand.

> **CONFIRMED BY EDA:** The dataset does NOT provide a direct stockout label of any kind. There is no inventory field.

> **PROPOSED / HYPOTHESIS — NOT YET PROVEN:** Therefore we cannot definitively identify stockouts. We can only investigate indirect signals (such as the price-history alignment in Finding 2) that may suggest an unusual zero period — and any conclusion drawn from those signals remains a hypothesis, not a proven fact.

## Promotion and Event Handling

> **CONFIRMED BY EDA:** A full column-by-column scan of all five files found no promotion, discount, markdown, or “on-deal” flag anywhere in the dataset.

Because of this, we should not simply label every sales spike as “a promotion happened.” Instead, the plan is to investigate several indirect, imperfect signals together:

- Named events and their specific identity (not just “was there an event, yes/no”)
- Event type categories (Sporting, Cultural, National, Religious)
- Price changes, used cautiously (see Finding 9 — not automatically treated as a promotion)
- Recurring calendar patterns (weekday, month, SNAP)
- Unusual deviations from a series' normal demand level

> **Term explained — Demand shock:** an unusually large increase or decrease in sales relative to a product's normal behavior, from any cause.

> **Why this matters:** Finding 8 showed that named events move sales in opposite directions depending on which event it is (Christmas −99.95% vs. Labor Day +27.5%). This is exactly why event-aware forecasting means letting the model learn that different events have different effects, rather than assuming every event boosts sales the same way, or at all.

## Novelty Statement

> **PROPOSED / HYPOTHESIS — NOT YET PROVEN:** THE NOVELTY IS A HYPOTHESIS THAT MUST BE EXPERIMENTALLY VALIDATED. If the baseline performs better, we must reconsider or drop the novelty.

Our proposed novelty, “Listing-Aware Hurdle Forecasting with Recency State,” is not simply “we used LightGBM.” LightGBM is just the practical tool. The actual novelty is the combination of five specific modeling choices, each tied directly to a specific EDA finding:

1. Treating sale occurrence separately from sale magnitude (a hurdle / two-stage structure), responding to the 68% zero-inflation found in Finding 1.
2. Accounting for possible pre-listing / inactive periods, responding to the leading-zero / leading-no-price alignment found in Finding 2.
3. Explicitly modeling recency / dry-spell behavior (days_since_last_sale, zero_streak_length), responding to the near-perfect staircase relationship found in Finding 4.
4. Using future-known context — calendar, named events, SNAP, and price — that is already available for the entire forecast horizon, responding to Findings 6–9.
5. Testing, rather than assuming, whether each of these choices actually improves 28-day forecasting accuracy (see the Experiment Plan below).

> **Why this matters:** Item 5 is arguably the most important part of the novelty. A method is only genuinely useful if it is shown to help — not just because it sounds more sophisticated.

## Experiment Plan

To find out which components actually help, rather than assuming they all do, we plan a staged sequence of experiments, each adding one component on top of the last:

| # | Experiment | What it adds |
|---|---|---|
| 1 | Simple baseline | Seasonal-naive or an equivalent simple reference forecast |
| 2 | Single LightGBM | One combined model with standard engineered features (Groups A–I) |
| 3 | Two-stage hurdle model | Splits occurrence (Stage 1) from magnitude (Stage 2) |
| 4 | + Listing-aware features | Adds the pre-listing / activity indicator on top of Experiment 3 |
| 5 | + Recency-state features | Adds days_since_last_sale / zero_streak_length on top of Experiment 4 |

Comparing the validation performance across Experiments 1→5 tells us which specific component is actually responsible for any improvement — this is called an ablation study.

> **Term explained — Ablation study:** an experiment where one component is removed (or added) at a time, to determine whether that specific component is actually responsible for a change in performance.

> **Example:** Suppose the full model (Experiment 5) shows a 10% improvement over the baseline. If we then remove just the recency features and performance drops to only a 4% improvement, we can conclude the recency features themselves were responsible for roughly 6 percentage points of that gain — a meaningful, specific contribution, not just a vague overall improvement.

## Metrics

> **PROPOSED / HYPOTHESIS — NOT YET PROVEN:** The exact official competition/hackathon scoring metric must be confirmed before final implementation. We are NOT inventing or assuming it here.

The M5 competition this dataset comes from commonly uses a metric called WRMSSE, but this is general background knowledge about that public competition, not something confirmed against our own hackathon's instructions — it is explicitly labeled here as needing confirmation.

> **Term explained — WRMSSE:** Weighted Root Mean Squared Scaled Error — the metric commonly used in the public M5 competition. It weights errors by each series' sales volume and scales them against that series' own historical variability. Needs confirmation for this hackathon specifically.

Until the official metric is confirmed, we can use standard, well-understood validation metrics to compare experiments internally:

| Metric | In plain English |
|---|---|
| MAE (Mean Absolute Error) | The average size of the forecast error, ignoring whether it was too high or too low. |
| RMSE (Root Mean Squared Error) | Similar to MAE, but penalizes large errors more heavily than small ones. |
| WAPE (Weighted Absolute Percentage Error) | Total absolute error as a percentage of total actual sales — useful for comparing across series of very different sizes, if appropriate for our use. |

## Final 28-Day Forecast Process

This section describes the process to follow after validation and model selection are complete — it is documented here for planning purposes only and has not been executed.

1. Train the chosen model using the appropriate historical data.
2. Generate features using only information available at the forecast origin (per the leakage rules agreed in Section 10).
3. Predict the 28-day horizon: 2016-05-23 → 2016-06-19.
4. Generate predictions for all 30,490 store-item series.
5. Format predictions to match the exact structure required by sample_submission.csv.
6. Perform final sanity checks (no negative predictions, correct row/column shape, no missing values).
7. Never overwrite the original dataset files.

## Project Pipeline

The complete pipeline, from raw data to final submission, showing where this document sits:

```
RAW DATASET
    |
    v
DATA VALIDATION / PROCESSING  (complete — see PROCESSING_REPORT)
    |
    v
PROCESSED DATASET  (sales_long_full.parquet)
    |
    v
EDA  (complete — see EDA_REPORT)
    |
    v
TEAM DISCUSSION  ← THIS DOCUMENT SUPPORTS THIS STEP
    |
    v
FEATURE ENGINEERING  (not yet started)
    |
    v
BASELINE  (not yet built)
    |
    v
HURDLE MODEL  (not yet built)
    |
    v
BACKTESTING  (not yet run)
    |
    v
ABLATION / NOVELTY TESTING  (not yet run)
    |
    v
MODEL SELECTION  (not yet made)
    |
    v
FINAL 28-DAY FORECAST  (not yet generated)
    |
    v
SUBMISSION  (not yet made)
```

## Team Decisions Required

The following decisions need to be made collaboratively by the team before implementation begins. None of them have been decided yet by this document.

1. Do we use the hurdle / two-stage approach, or a single combined model?
2. How exactly do we define “listing / activity” for the listing-aware features?
3. Which candidate features (Section 11) are approved for the first build?
4. Recursive vs. direct/fixed-origin forecasting — which leakage-safe strategy do we use?
5. Single global model vs. segment-specific models (e.g., by activity class)?
6. What is the exact evaluation metric for this hackathon?
7. Which novelty components (Section 14) are actually worth keeping, based on the ablation results?
8. How will we detect and handle suspicious / unusual zero periods?
9. How will we treat price changes — as a feature, and how cautiously?
10. How many historical 28-day backtest windows will we use, and which ones?

## Glossary

Every technical term used in this document, collected in one place.

| Term | Plain-English meaning |
|---|---|
| Zero-inflation | A dataset where far more values are exactly zero than a typical smooth distribution would predict — 68.0% of our rows. |
| Intermittent demand | A pattern where a product sells on some days but has zero sales on many other days. |
| Classification | A model that predicts a category or outcome (e.g., yes/no) rather than a numeric quantity. |
| Regression | A model that predicts a numeric quantity directly. |
| Hurdle model | A two-stage approach: first predict whether an event happens (occurrence), then predict its size given that it happens (magnitude). |
| Lag feature | A previous day's or week's sales value used as a model input (e.g., lag_7 = sales 7 days ago). |
| Rolling feature / rolling mean | A feature summarizing recent history, such as the average sales over the previous 7 or 28 days. |
| Feature engineering | The process of building input variables (features) for a model from raw data. |
| LightGBM | A fast, tree-based machine-learning algorithm, practical for large datasets like ours. |
| Loss function | The mathematical method used to measure how wrong a model's predictions are during training. |
| Tweedie | A loss/distribution family suited to non-negative, skewed data with many zeros; still needs testing on our data. |
| Leakage (data leakage) | When a model accidentally uses information that would not actually be available at the real moment a forecast is made. |
| Backtesting | Pretending an earlier point in history is “today,” forecasting forward from it, and checking against sales we already know happened. |
| Baseline | A simpler model or method used as a reference point to judge whether a more complex approach is actually worth it. |
| Ablation study | An experiment where one component is added or removed at a time, to find out whether that specific component helps. |
| WRMSSE | Weighted Root Mean Squared Scaled Error — the metric commonly used in the public M5 competition; needs confirmation for this hackathon. |
| SNAP | Supplemental Nutrition Assistance Program — a US food-assistance benefit; the dataset flags whether it was usable per state, per day. |
| Demand shock | An unusually large increase or decrease in sales relative to a product's normal behavior. |
| days_since_last_sale | A feature counting how many days have passed since a store-item combination last recorded a sale. |

## Appendix A — Source Documents Examined

This document was prepared using only the following existing project materials. No new dataset analysis was performed to write this plan — all figures cited above trace back to one of these sources.

- DATASET_SUMMARY.md — the original dataset investigation (files, schema, relationships, quality checks)
- DATASET_EXPLAINED.pdf — the study-guide version of the dataset investigation
- PROCESSING_REPORT (pdf) — documents the raw-to-long-format join pipeline and the resulting sales_long_full.parquet
- EDA_REPORT (pdf) — documents distribution, zero-sales, seasonality, event, SNAP, price, heterogeneity, and correlation findings, plus candidate features and leakage considerations
- PS11_Walkthrough_Simple_Updated.docx — the plain-language problem-statement walkthrough, used for project-context framing only

> **NOTE:** This document was generated in an environment that does not itself contain the team's raw_dataset/ or processed_dataset/ folders — only the four report documents listed above were available. No file inside raw_dataset/ or processed_dataset/ was read, written, or modified while preparing this plan, because this environment never had access to them in the first place. See _audit/AUDIT_LOG.md for the full detail.

## Appendix B — Confirmed Findings vs. Proposed Hypotheses

A single-glance summary of what is settled fact (from our own EDA) versus what this document proposes and still needs to test. This document does not train a model, does not create the final feature dataset, and does not make a final prediction — it exists solely so the team can review and agree on the modeling strategy above before implementation begins.

| Statement | Status |
|---|---|
| 68.0% of all daily observations are zero sales | CONFIRMED (EDA) |
| rolling_mean_7 (r=0.820) is the strongest same-day predictor found | CONFIRMED (EDA) |
| days_since_last_sale shows a near-perfect staircase with P(sale) | CONFIRMED (EDA) |
| Leading zero-sales gaps align with leading no-price gaps (99.48% within 7 days) | CONFIRMED (EDA) |
| Weekend sales are 31.1% higher than weekday sales | CONFIRMED (EDA) |
| SNAP days show 12.7% higher mean sales overall, concentrated in FOODS | CONFIRMED (EDA) |
| Named events have large, opposite-direction effects (Christmas vs. Labor Day) | CONFIRMED (EDA) |
| No promotion flag or stockout flag exists anywhere in the dataset | CONFIRMED (EDA) |
| Calendar, price, and SNAP are known in advance for the full 28-day forecast window | CONFIRMED (EDA) |
| Leading zero-sales periods represent products “not yet listed” | HYPOTHESIS — well-supported, not proven |
| A two-stage hurdle model will outperform a single combined model | HYPOTHESIS — to be tested (Experiment 3) |
| Listing-aware features will meaningfully improve accuracy | HYPOTHESIS — to be tested (Experiment 4) |
| Recency-state features will meaningfully improve accuracy on top of the above | HYPOTHESIS — to be tested (Experiment 5) |
| Price changes can serve as a weak, indirect demand-shock signal | HYPOTHESIS — no promotion field to confirm causally |
| LightGBM with Tweedie loss is the best model choice | PROPOSED CANDIDATE — not a final decision |
| The official scoring metric is WRMSSE | UNCONFIRMED — needs verification with organizers |
