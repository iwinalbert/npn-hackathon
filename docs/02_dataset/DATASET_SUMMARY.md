# Dataset Summary — Problem Statement 11 (Retail Demand Forecasting)

**Investigation scope:** `C:\Users\Rishi\OneDrive\Desktop\NPN_HACKATHON\DATASET_m5-forecasting-accuracy\`
**Method:** Every number in this report was computed directly from the actual CSV files using pandas/numpy (see `analysis_output/` for the raw stats, aggregate CSVs and charts this report is built from). Nothing here is copied from general knowledge of the public M5 Kaggle competition — where the files happen to match well-known M5 conventions, that is stated as an *observation*, not assumed in advance.
**No original file was modified.** All derived files live in `analysis_output/`.

---

## 0. Explain Like I'm a Student (read this first, ~3 min)

We have **5 files** in `DATASET_m5-forecasting-accuracy/`:

1. **`sales_train_evaluation.csv`** (116 MB) — the main file. Each row is one **product sold in one store** (30,490 such combinations), and each column is one **day**. There are 1,941 daily columns (`d_1` … `d_1941`). A cell = how many units of that product were sold in that store on that day.
2. **`sales_train_validation.csv`** (114 MB) — the exact same 30,490 products/stores, but only going up to day 1,913 (28 fewer days). It looks like an earlier snapshot of the same data — everything in it is also contained in the evaluation file.
3. **`calendar.csv`** (0.1 MB) — one row per calendar date (1,969 dates, from **2011-01-29 to 2016-06-19**), telling us the weekday, month, year, any special event that day, and whether SNAP (a US food-assistance benefit) is active that day in California, Texas, or Wisconsin.
4. **`sell_prices.csv`** (194 MB, 6.84M rows) — the weekly selling price of every product in every store.
5. **`sample_submission.csv`** (5 MB) — a template of what a submission should look like: one row per product-store, 28 forecasted values (`F1`...`F28`), all currently zero.

**What we're predicting:** for each of the 30,490 store-product combinations, the number of units that will sell on each of the next 28 days.

**How much history do we have:** ~5.3 years of daily sales (1,941 days, no gaps). That's a lot — enough to learn weekly and yearly seasonality with confidence.

**What's available about the future:** here is the most important finding in this whole investigation. `calendar.csv` runs 28 days **past** the last day of actual sales data (`d_1942` to `d_1969`, i.e. 2016-05-23 to 2016-06-19), and `sell_prices.csv` also has price data for that exact same future window. That means **calendar features (weekday, month, events, SNAP) and prices for the forecast horizon are already known and provided** — we don't have to guess them. We just don't have actual *sales* for those 28 days, which is exactly what we need to predict.

**What makes this hard:**
- **68% of all daily observations are zero.** Most products don't sell every day in every store — this is "intermittent demand," which breaks a lot of standard forecasting methods.
- There's no explicit "promotion" flag anywhere in the dataset — only price, so any promotional effect has to be inferred indirectly from price drops.
- Products get added to stores' assortments partway through the timeline (a product can have zero sales for its first N days simply because it wasn't being sold yet, not because demand was zero) — this needs to be recognized, not treated as ordinary intermittency.

**Biggest uncertainty:** whether the 28-day evaluation window in `sales_train_evaluation.csv` (days 1914–1941, which we now have actual values for) should be treated as extra training signal or held out as a validation set — that's a modeling decision for later, not something the data itself resolves.

**What to investigate next:** exact feature engineering plan (lag/rolling features, price-change signal, event proximity), and how to handle the "product not yet listed" zeros vs. genuine stockouts/no-demand zeros.

---

## 1. Dataset Files

| File | Rows | Columns | Size | Purpose | Important columns |
|---|---|---|---|---|---|
| `calendar.csv` | 1,969 | 14 | 0.10 MB | Maps every `d_` day index to a real calendar date, plus events & SNAP flags | `date`, `d`, `wm_yr_wk`, `event_name_1/2`, `event_type_1/2`, `snap_CA/TX/WI` |
| `sales_train_validation.csv` | 30,490 | 1,919 (6 id + 1,913 day) | 114.45 MB | Daily unit sales per store-item, days `d_1`–`d_1913` | `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`, `d_1`...`d_1913` |
| `sales_train_evaluation.csv` | 30,490 | 1,947 (6 id + 1,941 day) | 116.10 MB | Same as above but extended 28 more days, `d_1`–`d_1941` (superset of validation) | same as above, `d_1`...`d_1941` |
| `sell_prices.csv` | 6,841,121 | 4 | 193.97 MB | Weekly selling price per store-item | `store_id`, `item_id`, `wm_yr_wk`, `sell_price` |
| `sample_submission.csv` | 60,980 | 29 | 4.99 MB | Submission template — 28-day forecast per store-item, two ID blocks (`_validation`, `_evaluation`) | `id`, `F1`...`F28` |

Total dataset size on disk: **≈ 430 MB**. Total lines across all files: **≈ 6.97 million**.

**How the files relate (facts, verified):**
- `sales_train_validation.csv` and `sales_train_evaluation.csv` contain **identical** sets of 30,490 base IDs, in identical row order — confirmed by string comparison after stripping the `_validation`/`_evaluation` suffix. The evaluation file simply has 28 more day-columns.
- `calendar.d` (`d_1`...`d_1969`) is the shared time key joining `calendar.csv` to the day-columns of both sales files.
- `calendar.wm_yr_wk` is the shared key joining `calendar.csv` to `sell_prices.csv` (prices are weekly, not daily).
- `sell_prices.csv` keys on (`store_id`, `item_id`), matching the sales files' `store_id`/`item_id` columns.
- `sample_submission.csv`'s `id` column, once suffixes are stripped, is an exact match (verified by set equality) to both the sales files' and price files' store-item universe.

---

## 2. Dataset Schema

### 2.1 `calendar.csv` (1,969 rows × 14 columns)

| Column | Type | Meaning | Example | Unique | Missing | Missing % | Min/Max |
|---|---|---|---|---|---|---|---|
| `date` | date (string→datetime) | Calendar date | `2011-01-29` | 1,969 | 0 | 0% | 2011-01-29 / 2016-06-19 |
| `wm_yr_wk` | int | Walmart-internal year-week code, joins to `sell_prices` | `11101` | 282 | 0 | 0% | 11101 / 11621 |
| `weekday` | string | Day name | `Saturday` | 7 | 0 | 0% | — |
| `wday` | int | Day-of-week index, 1=Saturday...7=Friday | `1` | 7 | 0 | 0% | 1 / 7 |
| `month` | int | Calendar month | `1` | 12 | 0 | 0% | 1 / 12 |
| `year` | int | Calendar year | `2011` | 6 | 0 | 0% | 2011 / 2016 |
| `d` | string | Day index used as the sales-file column name | `d_1` | 1,969 | 0 | 0% | d_1 / d_1969 |
| `event_name_1` | string | Name of a special event on this date, if any | `SuperBowl` | 30 | 1,807 | 91.8% | — |
| `event_type_1` | string | Category of `event_name_1` | `Sporting` | 4 (`Sporting`,`Cultural`,`National`,`Religious`) | 1,807 | 91.8% | — |
| `event_name_2` | string | A second, simultaneous event (rare) | `Easter` | 4 | 1,964 | 99.7% | — |
| `event_type_2` | string | Category of `event_name_2` | `Cultural` | 2 (`Cultural`,`Religious`) | 1,964 | 99.7% | — |
| `snap_CA` | int (0/1) | Whether SNAP benefits are usable in CA that day | `1` | 2 | 0 | 0% | 0 / 1 |
| `snap_TX` | int (0/1) | Same, Texas | `0` | 2 | 0 | 0% | 0 / 1 |
| `snap_WI` | int (0/1) | Same, Wisconsin | `0` | 2 | 0 | 0% | 0 / 1 |

**What this means for forecasting:** `event_name_1`/`event_type_1` is a genuine (if sparse — only 162 of 1,969 days, 8.2%, have a primary event) known-in-advance calendar signal — Super Bowl, Mother's/Father's Day, religious holidays, sporting-season markers, etc. `event_name_2` fires on only 5 dates total (days where two events coincide, e.g. Easter + Orthodox Easter in 2014) and is essentially negligible on its own. `snap_*` is a clean deterministic binary calendar rule, present with **zero missing values across all 1,969 rows including the 28 future days** — verified, not assumed.

### 2.2 `sales_train_evaluation.csv` (30,490 rows × 1,947 columns) — primary sales fact table

| Column | Type | Meaning | Example | Unique | Missing |
|---|---|---|---|---|---|
| `id` | string | `{item_id}_{store_id}_evaluation`, unique row key | `HOBBIES_1_001_CA_1_evaluation` | 30,490 | 0 |
| `item_id` | string | Product identifier | `HOBBIES_1_001` | 3,049 | 0 |
| `dept_id` | string | Department | `HOBBIES_1` | 7 | 0 |
| `cat_id` | string | Category (top of product hierarchy) | `HOBBIES` | 3 | 0 |
| `store_id` | string | Store identifier | `CA_1` | 10 | 0 |
| `state_id` | string | State | `CA` | 3 | 0 |
| `d_1` ... `d_1941` | int | Units sold that day, that store-item | `0`, `3`, `130` | — | **0 across all 59,181,090 cells** |

`sales_train_validation.csv` has the identical schema, just `d_1`...`d_1913` and `_validation` suffix on `id`.

**No missing values were found anywhere in the day columns of either sales file** — every one of the 59,181,090 evaluation cells (30,490 × 1,941) is a non-null integer ≥ 0.

### 2.3 `sell_prices.csv` (6,841,121 rows × 4 columns)

| Column | Type | Meaning | Example | Unique | Missing | Min/Max |
|---|---|---|---|---|---|---|
| `store_id` | string | Store | `CA_1` | 10 | 0 | — |
| `item_id` | string | Product | `HOBBIES_1_001` | 3,049 | 0 | — |
| `wm_yr_wk` | int | Week key (joins `calendar.wm_yr_wk`) | `11325` | 282 | 0 | 11101 / 11621 |
| `sell_price` | float | Selling price that week, that store-item | `9.58` | — | 0 | **$0.01 / $107.32** |

No zero or negative prices found. Every (`store_id`,`item_id`) pair present in the sales files has **at least one** price row (0 pairs missing entirely — checked against the full 30,490).

### 2.4 `sample_submission.csv` (60,980 rows × 29 columns)

| Column | Type | Meaning |
|---|---|---|
| `id` | string | Store-item id, `_validation` (30,490 rows) or `_evaluation` (30,490 rows) suffixed |
| `F1`...`F28` | int, currently all 0 | Placeholder for the 28-day forecast |

Both the `_validation`-suffixed and `_evaluation`-suffixed ID sets, after stripping suffixes, are **identical** to each other and to the sales files' 30,490 store-item universe (verified by set equality). This means the competition/task expects **two separate 28-day forecasts per store-item**: one that would correspond to days `d_1914`–`d_1941` (already revealed as real sales in `sales_train_evaluation.csv`) and one for days `d_1942`–`d_1969` (the true future — see Section 3).

---

## 3. Date and Time Analysis

Timeline (all values read directly from `calendar.csv` and the `d_` column counts of the two sales files):

```
Historical sales start:  2011-01-29  (d_1)
        ↓  1,941 continuous days, no gaps, no duplicate dates
Last observed sale:      2016-05-22  (d_1941, sales_train_evaluation.csv)
        ↓
"Validation window" (already revealed): d_1914–d_1941 = 2016-04-25 → 2016-05-22
   (this is the gap between sales_train_validation's last day, d_1913 = 2016-04-24,
    and sales_train_evaluation's last day)
        ↓
Potential forecasting period:  d_1942 – d_1969 = 2016-05-23 → 2016-06-19  (28 days)
        ↓
Future information available for this window?
   calendar.csv → YES, all 28 days present (weekday, month, events, SNAP)
   sell_prices.csv → YES, prices exist up to wm_yr_wk 11621, which covers this window
   sales → NOT AVAILABLE — this is exactly what must be predicted
```

Key facts:
- **Earliest date:** 2011-01-29. **Latest date (in `calendar.csv`):** 2016-06-19.
- **1,969 unique dates**, **1,969 rows** — one row per date, no duplicates.
- **Continuity check:** generating a full daily date range between min and max produces exactly 1,969 days, matching the row count exactly → **zero missing dates, fully continuous**.
- **Historical sales period:** 1,941 days (`sales_train_evaluation.csv`) ≈ **5.31 years**. (`sales_train_validation.csv` covers a shorter 1,913 days ≈ 5.24 years — it is a strict prefix of the evaluation file's timeline.)
- **This is far more than "two months"** — nearly 5.5 years of daily history are present.
- **Future dates confirmed:** `calendar.csv` extends 28 days beyond the last day of actual sales in `sales_train_evaluation.csv`. This is not an assumption — it's a direct row-count fact (1,969 calendar rows vs. 1,941 sales-day columns).
- `wday`/`weekday` confirms a consistent 7-day cycle (`wday=1`→Saturday ... `wday=7`→Friday) with no irregularities across the full range.

---

## 4. Understanding the Target

**Target column:** the `d_1`, `d_2`, ..., `d_1941` columns in the sales files. Each is an integer count of **units sold**, not revenue (no dollar sign, and cross-referencing with `sell_prices` confirms these are raw unit counts that get multiplied by price only if revenue is needed downstream — that multiplication is not done anywhere in the provided files).

**Granularity: one target value = one (item, store, day) triple.** There is a single target definition, not multiple targets — every row is a full daily time series for one store-item pair, and the columns are consecutive days.

**Concrete example, using real values from the file:**

> `id = FOODS_3_090_CA_3_evaluation` → `item_id = FOODS_3_090`, `store_id = CA_3` → on day `d_1` (2011-01-29) this row's value is its `d_1` cell (part of a series whose 1,941-day total is **253,859 units**, the highest-volume series in the whole dataset).

So: **Item `FOODS_3_090` + Store `CA_3` + Date `2011-01-29` → Sales = (value in that cell).** The forecasting unit is confirmed to be **store × item**, at daily granularity, for all 3,049 items × 10 stores = **30,490 independent series**.

**Hierarchical aggregation levels exist as metadata** (see Section 7) — `item_id` rolls up into `dept_id` → `cat_id`, and `store_id` rolls up into `state_id` → total — but the file that must actually be filled in (`sample_submission.csv`) is defined **only at the store-item level**; there is no separate file asking for state-level or category-level forecasts directly.

---

## 5. Store / Location Analysis

- **10 stores**, IDs: `CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3`.
- **3 states**: California (4 stores), Texas (3 stores), Wisconsin (3 stores) — mapping verified directly from `state_id` vs `store_id` in the sales file (`CA→[CA_1..CA_4]`, `TX→[TX_1..TX_3]`, `WI→[WI_1..WI_3]`).
- **Every store carries the identical 3,049-item catalog** — verified: each item appears in exactly 10 stores, no exceptions, so the store-item panel is a perfectly balanced 3,049 × 10 grid (30,490 rows, matches exactly).
- **Records per store:** 3,049 series × 1,941 days = 5,919,109 daily observations, identical for every store (this is a structural fact, not a sampling artifact).

**Total units sold per store, full history (from `analysis_output/store_summary.csv`):**

| Store | Total units sold | Avg daily units/item | % zero-sale days |
|---|---|---|---|
| CA_3 | 11,363,540 | 1.92 | 59.4% |
| CA_1 | 7,832,248 | 1.32 | 63.8% |
| TX_2 | 7,329,642 | 1.24 | 66.3% |
| WI_2 | 6,697,988 | 1.13 | 70.5% |
| WI_3 | 6,542,557 | 1.11 | 70.0% |
| TX_3 | 6,205,940 | 1.05 | 69.7% |
| CA_2 | 5,818,395 | 0.98 | 68.8% |
| TX_1 | 5,692,823 | 0.96 | 70.7% |
| WI_1 | 5,261,506 | 0.89 | 68.8% |
| CA_4 | 4,182,534 | 0.71 | 72.0% |

**Same product behaves very differently across stores.** Example: `FOODS_3_090` is the #1 seller in `CA_3` (253,859 units, only 18.5% zero-days) but sells markedly less — though still in the top 10 overall — in `CA_1`, `WI_3`, `TX_2`, `TX_3` (each ~117k–129k units, ~24–25% zero-days). `CA_3` alone (1 of 10 stores) accounts for **17% of total company-wide units sold**, more than 2.7× the lowest-volume store (`CA_4`). This is a real, store-specific effect visible directly in the aggregated data, not an assumption.

Chart: `analysis_output/04_sales_by_store.png`

---

## 6. Product Analysis

- **3,049 unique products** (`item_id`), organized into **7 departments** (`dept_id`) under **3 categories** (`cat_id`).
- Product hierarchy (verified via `groupby`): `FOODS → {FOODS_1, FOODS_2, FOODS_3}`, `HOBBIES → {HOBBIES_1, HOBBIES_2}`, `HOUSEHOLD → {HOUSEHOLD_1, HOUSEHOLD_2}`.

**Items and total sales per category:**

| Category | # items (×10 stores = rows) | Total units sold | Avg daily units/item | % zero-sale days |
|---|---|---|---|---|
| FOODS | 1,437 items (14,370 rows) | 45,922,427 | 1.65 | 61.8% |
| HOUSEHOLD | 1,047 items (10,470 rows) | 14,764,090 | 0.73 | 71.6% |
| HOBBIES | 565 items (5,650 rows) | 6,240,656 | 0.57 | 77.1% |

**Per department:**

| Department | # rows | Total units sold | Avg daily units/item | % zero-sale days |
|---|---|---|---|---|
| FOODS_3 | 8,230 | 32,937,002 | 2.06 | 58.6% |
| HOUSEHOLD_1 | 5,320 | 11,722,853 | 1.14 | 62.9% |
| HOBBIES_1 | 4,160 | 5,699,014 | 0.71 | 73.1% |
| FOODS_2 | 3,980 | 7,795,025 | 1.01 | 67.8% |
| HOUSEHOLD_2 | 5,150 | 3,041,237 | 0.30 | 80.6% |
| FOODS_1 | 2,160 | 5,190,400 | 1.24 | 63.0% |
| HOBBIES_2 | 1,490 | 541,642 | 0.19 | 88.4% |

FOODS dominates volume (69% of all units sold company-wide) despite being only 47% of item-store rows. `HOBBIES_2` is the weakest department by a wide margin — lowest per-item average and 88.4% zero-sale days.

**Extremes (from `analysis_output/series_summary.csv`, all 30,490 store-item series ranked by total units sold over the full history):**

Top 5: `FOODS_3_090_CA_3` (253,859), `FOODS_3_586_TX_2` (195,120), `FOODS_3_586_TX_3` (151,862), `FOODS_3_586_CA_3` (136,269), `FOODS_3_090_CA_1` (128,855).

Bottom 5 (lowest nonzero totals — every series has *some* sales, see Section 8): `FOODS_3_778_CA_2` (15 units total, 99.4% zero-days), `FOODS_2_057_WI_2` (16 units), `HOBBIES_1_170_WI_3` (16 units), `FOODS_2_071_TX_3` (18 units), `HOUSEHOLD_1_378_CA_1` (23 units).

Chart: `analysis_output/05_sales_by_department.png`

---

## 7. Hierarchy Analysis

The dataset contains exactly these hierarchy levels, confirmed directly from the sales-file columns (nothing more, nothing less):

```
item_id        (3,049 levels)
   ↓
dept_id        (7 levels)
   ↓
cat_id         (3 levels)

store_id       (10 levels)
   ↓
state_id       (3 levels)

Total          (1 level — implicit, i.e. sum over everything)
```

Both the product hierarchy (item → dept → cat) and the location hierarchy (store → state) are **present as metadata columns on every sales row** — they are not separate lookup files that need joining; `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` all sit directly on `sales_train_validation.csv`/`sales_train_evaluation.csv`.

**Is there a combined item×store cross hierarchy?** Yes — every item is sold in every store (verified: min/max stores-per-item = 10/10 for all 3,049 items), so `item_id × store_id` forms a complete 30,490-cell grid, and levels like `item × state`, `dept × store`, `cat × state`, `dept × state`, etc. can all be derived by aggregation without any missing combinations.

**Is hierarchy just metadata, or must forecasts be coherent across it?** Based purely on the files: `sample_submission.csv` only asks for forecasts **at the store-item level** — there is no separate submission slot for department, category, state, or total-level forecasts. So structurally, the required deliverable is bottom-level only. Whether a *modeling approach* additionally enforces coherence with higher levels (e.g. reconciliation, as hinted at in the problem statement) is a methodology choice for later, not something dictated by the files themselves.

---

## 8. Sales / Demand Analysis

Computed over all 59,181,090 cells of `sales_train_evaluation.csv` (30,490 series × 1,941 days):

| Statistic | Value |
|---|---|
| Mean | 1.131 |
| Median | 0.0 |
| Std dev | 3.870 |
| Min | 0 |
| Max | 763 (a single-day spike) |
| P25 | 0 |
| P75 | 1 |
| P90 | 3 |
| P95 | 5 |
| P99 | 15 |
| P99.9 | 47 |
| Zero-sale observations | **40,241,819 (68.00%)** |
| Negative values | **0** |
| Missing values | **0** |

**Intermittent demand is the dominant pattern in this dataset** — 68% of every store-item-day observation is exactly zero, and the median is 0. This is not a data-quality artifact: negative values and missing values are both exactly zero, confirming the zeros are real "nothing sold that day," not corrupted data.

**At the series level** (per store-item, over its full 1,941-day history):
- **0 of 30,490 series are permanently dead** (every series has *some* nonzero sales at some point).
- **0 series have 0% zero-days** — literally every single product-store combination has at least one zero-sale day; a purely continuous, always-selling series does not exist in this dataset.
- **1,534 series (5.0%) have >95% zero-days** — near-dormant products.
- **23,852 series (78.2%) have >50% zero-days** — the intermittent pattern is the *majority* case, not an edge case.
- Median zero-days-per-series: **73.3%**.

**Skew:** the distribution is extremely right-skewed — mean (1.13) is far above the median (0), and the max single observation (763) is over 600× the mean. A handful of high-volume FOODS_3 series drive a disproportionate share of total volume (Section 6).

**Outliers / spikes:** the max value of 763 and the P99.9 of 47 indicate rare but large single-day spikes exist on top of typical single-digit daily sales — worth investigating against the event calendar later (a sample check found average CA `FOODS` sales are ~10% higher on SNAP days than non-SNAP days — see `analysis_output/09_snap_effect.png` — but a rigorous event/spike correlation study is future work, not done here).

**What this means for forecasting:** standard regression losses (MSE) and models that assume roughly-continuous, non-negative-skewed targets will struggle here. This is a textbook **intermittent/count demand** problem — approaches built for that (e.g. Croston-style methods, quantile/count-aware losses, tree-based models with zero-inflation awareness) are typically more appropriate than naive continuous-time-series methods. (Noting this as a data characteristic — **not** selecting an algorithm, per instructions.)

Charts: `analysis_output/02_sales_distribution.png`, `analysis_output/03_zero_sales_distribution.png`

---

## 9. Time-Series Behavior

All of the following was read off the actual daily aggregates in `analysis_output/daily_total_sales.csv`, `store_daily_sales.csv`, and `cat_daily_sales.csv` — not assumed.

- **Trend:** Total daily units sold (summed across all 30,490 series) show a **clear upward trend** over the full history — the 28-day rolling average rises from roughly ~25,000–30,000 units/day in 2011 to roughly ~40,000–45,000 units/day by 2016. See `analysis_output/01_overall_sales_over_time.png`.
- **Weekly pattern:** a strong 7-day cycle is visible in the raw daily series (the thin blue "daily total" line in the same chart oscillates regularly). Aggregating average total sales by `weekday` (`analysis_output/07_weekly_pattern.png`) shows **weekends (Saturday, Sunday) are the highest-volume days**, consistent across the whole history.
- **Monthly/seasonal patterns:** with 5.3 years of history spanning 6 different calendar years (2011–2016, all with essentially full-year coverage except the first (337 days from Jan 29) and last (171 days through June)), there is **enough history to examine yearly seasonality with real confidence** — multiple full years exist to compare month-over-month, unlike a 2-month dataset. (This report does not go further into decomposing seasonal components — that is deferred to the feature-engineering stage.)
- **Sudden spikes:** visible in the example series chart (`analysis_output/06_example_series.png`) — e.g., the top-selling item `FOODS_3_090_CA_3` shows repeated sharp spikes to 400–760 units against a typical band of 100–250.
- **Long zero-sale stretches:** the same example chart shows a large leading block of exact zeros for `FOODS_3_090_CA_3` from the start of the dataset until roughly mid-2011 — cross-checked in Section 14, this coincides with the product not yet having a price entry (i.e., not yet stocked), not a demand collapse.
- **Store- and product-specific patterns:** confirmed different volume/intermittency profiles per store (Section 5) and per department (Section 6) — these are not uniform across the panel.

**What is NOT claimed:** this report does not claim multi-year seasonality has been *decomposed or quantified* (e.g., no STL decomposition or seasonal-strength metric was computed) — only that the data volume is sufficient to attempt it, which is a feasibility statement, not a finding about the seasonality's actual shape.

---

## 10. Calendar / Holiday / Event Data

Full inventory of calendar/event columns already given in Section 2.1. Key additional facts:

- **30 distinct named events** in `event_name_1` (e.g. SuperBowl, ValentinesDay, PresidentsDay, LentStart, StPatricksDay, Purim End, OrthodoxEaster, Pesach End, Cinco De Mayo, Mother's day, MemorialDay, NBAFinalsStart/End, Father's day, IndependenceDay, Ramadan starts, Eid al-Fitr, LaborDay, ColumbusDay, and others), grouped into **4 event types**: Sporting, Cultural, National, Religious.
- Events are sparse: only **162 of 1,969 days (8.2%)** have any `event_name_1`, and only **5 days** have a second simultaneous event.
- **SNAP flags** (`snap_CA`, `snap_TX`, `snap_WI`) are dense (0% missing, all 1,969 rows) and follow a deterministic, state-specific pattern: exactly **10 SNAP days per calendar month per state**, but the *specific* days differ by state (CA is always calendar days 1–10; TX and WI use a different, non-contiguous 10-day pattern each month) — this was verified programmatically, not assumed, and confirmed the three states' flags are **not** identical to one another.

### The critical question: can the model know about a future holiday/event?

> **"If I only have historical sales up to July and need to predict August, can the model know that an event/holiday occurs in August?"**

**Yes — confirmed directly from the files.** `calendar.csv` provides `event_name_1/2`, `event_type_1/2`, and `snap_CA/TX/WI` for **every one of the 1,969 dates, including the 28 days beyond the last day of actual sales data** (`d_1942`–`d_1969`, i.e. 2016-05-23 to 2016-06-19). None of these calendar/event columns had any missing values in that window. This is a genuine, dataset-provided future covariate — it does not need to be forecasted or assumed; it's already recorded in the file as ground truth about the calendar (holidays and SNAP eligibility dates are public, government/culturally-fixed schedules known well in advance of the date itself, which is presumably why they're already populated for the future window).

---

## 11. Price Analysis

All figures from `sell_prices.csv` (6,841,121 rows):

- **Coverage:** every one of the 30,490 store-item pairs has price data (verified: 0 pairs in the sales files are missing from the price file entirely).
- **Price range:** **$0.01 to $107.32**, mean **$4.41**. No zero or negative prices (0 rows with `sell_price ≤ 0`).
- **No missing prices** in the file (0/6,841,121 nulls in `sell_price`).
- **Price changes over time:** of the 30,490 store-item series, **8,247 (27.0%) have only a single price for their entire history** (never changed), while the rest show between 2 and 21 distinct price points over time (mean ≈ 2.77 distinct prices per series).
- **Coverage is not uniform across time per item** — the number of distinct weeks with a price for a given store-item ranges from **19 weeks to the full 282 weeks** (mean ≈ 224). Only 10,932 of 30,490 pairs (35.9%) have price data for literally every week in the dataset. The remaining pairs' price history starts partway through — this lines up with products being introduced to a store's assortment partway through the timeline (verified directly in Section 14).
- **Future prices are provided:** `sell_prices.csv`'s `wm_yr_wk` runs up to **11621**, exactly matching `calendar.csv`'s maximum `wm_yr_wk` (also 11621, corresponding to `d_1969`, the last future day). The last day of actual sales (`d_1941`) falls in week 11617 — so prices are available **4 weeks (28 days) past the last known sale**, i.e., **prices for the entire forecast horizon are already known**, not something that needs to be predicted or assumed constant.
- **Price vs. sales relationship:** a visual example for the top-selling series (`FOODS_3_090_CA_3`, `analysis_output/08_price_vs_sales.png`) shows price does step up and down over time (e.g. cycling between ~$1.00 and ~$1.44–$1.50 in 2014–2015) and these steps visually coincide with some of the same-period sales gaps — but this report does not establish a causal or even a robustly quantified correlation; it is flagged as an observation worth a dedicated price-elasticity analysis later, not a conclusion.

---

## 12. Promotion Analysis

**Not available in the provided dataset.** A full column-by-column scan of all 5 files found **no explicit promotion, discount, markdown, or "on-deal" flag or column anywhere** — the only price-related signal is `sell_price` in `sell_prices.csv` (a point-in-time selling price, not a promotion indicator). Consequently:

- Whether promotions exist: cannot be directly determined; only *inferred* indirectly from week-over-week price drops (see Section 11), which is a weak proxy since a price drop could reflect many things (regular repricing, competitive pricing, clearance, etc.) and isn't labeled as promotional in the data.
- Promotion frequency, duration, affected products/stores, and future promotion visibility: **Not available in the provided dataset** — there is nothing to measure.

This is an important limitation to carry forward: any "promotion effect" used in modeling would have to be **engineered from price-change patterns**, not read from a ground-truth promotion flag, since no such flag exists in these files.

---

## 13. Future Information / Forecasting Setup

| Feature | Historical available? | Future available? (d_1942–d_1969) | Known before prediction? | Can safely use for forecasting? |
|---|---|---|---|---|
| Past sales (`d_1`...`d_1941`) | Yes | No (this is the target) | N/A | Yes, as lag/rolling features only |
| Calendar (date, weekday, month, year) | Yes | **Yes** — all 28 future days present in `calendar.csv` | Yes (calendar is deterministic) | Yes |
| Day of week (`wday`/`weekday`) | Yes | **Yes** | Yes | Yes |
| Holiday/event (`event_name/type_1/2`) | Yes | **Yes** — verified non-missing for all 28 future days | Yes (public holiday calendar) | Yes |
| SNAP flag (`snap_CA/TX/WI`) | Yes | **Yes** — verified non-missing for all 28 future days | Yes (published benefit schedule) | Yes |
| Price (`sell_price`) | Yes | **Yes** — verified, price weeks extend to `wm_yr_wk` 11621 covering the full future window | Yes, as provided in this dataset | Yes, but note it's unusual to know exact future retail price this precisely in a live setting — here it's given, so usable |
| Promotion | **Not available in the provided dataset** | Not available | N/A | N/A — no such column exists to use |
| Store (`store_id`, `state_id`) | Yes | Yes (static attribute) | Yes | Yes |
| Product (`item_id`) | Yes | Yes (static attribute) | Yes | Yes |
| Category/Department (`cat_id`, `dept_id`) | Yes | Yes (static attribute) | Yes | Yes |
| `wm_yr_wk` (week code) | Yes | Yes | Yes | Yes (mainly as a join key) |

**Bottom line, directly from the files:** everything needed to build a feature set for the 28-day forecast horizon is available **except the target itself and any promotion signal**. This is a materially easier setup than a typical "pure time series, no covariates" forecasting problem — the calendar and price are true future-known covariates here, not leakage, because the files themselves supply them for the unlabeled window.

---

## 14. Data Quality

| Check | Result | Affects forecasting? | Needs cleaning? |
|---|---|---|---|
| Missing values in sales day-columns | **0 / 59,181,090 cells** (evaluation file) | No | No |
| Missing values in calendar (excl. sparse event columns) | 0 in date/weekday/month/year/wday/d/wm_yr_wk/snap_* | No | No |
| Missing `event_name_1/2`, `event_type_1/2` | 91.8% / 99.7% missing | Expected — most days have no event; must be encoded as "no event," not imputed | No cleaning needed, just correct encoding later |
| Missing values in `sell_prices.csv` | 0 / 6,841,121 | No | No |
| Duplicate rows in sell_prices (`store_id`,`item_id`,`wm_yr_wk`) | 0 | — | No |
| Duplicate `id` in either sales file | 0 | — | No |
| Duplicate dates in calendar | 0 | — | No |
| Negative sales values | 0 | — | No |
| Negative or zero prices | 0 | — | No |
| Date continuity (calendar) | Fully continuous, 1,969/1,969 days present | — | No |
| Broken relationships: sales↔calendar (`d` key) | All `d_1`...`d_1941` map to a valid calendar row | No | No |
| Broken relationships: sales↔prices (store-item pairs) | 0 store-item pairs in sales missing from prices entirely | No | No |
| Missing store-item combinations | 0 — perfectly balanced 3,049 × 10 grid | No | No |
| **Partial price history per store-item** | 19,558 of 30,490 pairs (64.1%) do **not** have a price for every week in the dataset — their price series starts later than week 1 | **Yes, significant** — this reflects products entering the store assortment mid-timeline. Verified example: item `FOODS_3_595` in store `CA_1` has 0 sales for all 1,841 days before its first priced week, then 126 units afterward. This "leading zero" block must be distinguished from genuine no-demand zeros. | Not a data error — but needs explicit handling (e.g., trimming pre-listing zeros or flagging them) during feature engineering, not now |
| Inconsistent category/department naming | None found — `cat_id`→`dept_id`→`item_id` hierarchy is perfectly nested with no orphans | No | No |
| `sales_train_validation.csv` vs `sales_train_evaluation.csv` overlap | Fully redundant for their shared date range (evaluation is a strict superset) | Decide which to use as the base table | Not cleaning, a usage decision |

**Overall:** this is an unusually clean dataset — the only real "issue" (and it's not a data-quality defect, it's a real-world business fact) is the mid-timeline product introduction pattern producing structural leading zeros for about two-thirds of the store-item series.

---

## 15. Relationships Between Files

```
sales_train_evaluation.csv  (30,490 store-item series × 1,941 days)  [primary fact table]
   │
   ├── item_id, dept_id, cat_id, store_id, state_id  → embedded directly (no join needed for hierarchy)
   │
   ├── d_1...d_1941  ──(join key: calendar.d)──►  calendar.csv
   │                                                 └── date, weekday, wday, month, year,
   │                                                     event_name/type_1/2, snap_CA/TX/WI, wm_yr_wk
   │
   └── (store_id, item_id) + calendar.wm_yr_wk ──►  sell_prices.csv
                                                       └── sell_price (weekly)

sales_train_validation.csv  — same schema/keys, strict prefix subset (d_1...d_1913)

sample_submission.csv  — id (store_id+item_id, both _validation and _evaluation suffix) → F1...F28 placeholders
```

**Primary keys:**
- `sales_train_*.csv`: `id` (= `item_id` + `store_id` + suffix), unique per row (verified, 0 duplicates).
- `calendar.csv`: `d` (also `date`), unique per row.
- `sell_prices.csv`: composite key (`store_id`, `item_id`, `wm_yr_wk`), unique (verified, 0 duplicates).
- `sample_submission.csv`: `id`, unique.

**Foreign / join keys:** `calendar.d` ↔ sales files' day-column names; `calendar.wm_yr_wk` ↔ `sell_prices.wm_yr_wk`; (`store_id`,`item_id`) shared between sales files and `sell_prices.csv`.

**Cardinality:** sales-to-calendar is many-to-one (many day-columns per row, one calendar row per day — effectively the sales table is already wide/pivoted by date). Sales-to-prices is many-to-many at the raw grain (one store-item has many weekly prices; one week has many store-items), resolved via the composite key.

---

## 16. Data Volume and Computational Requirements

- **Total records:** 59,181,090 daily sales observations (evaluation file) + 6,841,121 price records + 1,969 calendar rows + 60,980 submission placeholder rows ≈ **66.1 million records** across the dataset.
- **Number of time series:** 30,490 (store × item), each with 1,941 daily observations.
- **Records per time series:** 1,941 (uniform — every series has exactly the same number of days, since the table is a dense day-column matrix, not a sparse date-value log).
- **Measured memory footprint:** loading just the day-columns of `sales_train_evaluation.csv` as a NumPy `int32` array measured **236.7 MB** in memory (30,490 × 1,941 cells). The full `sell_prices.csv` with efficient dtypes (`category` for IDs, `int32`/`float32` for numeric) loads comfortably as well.
- **Total raw CSV size:** ≈ 430 MB.
- **Can this be processed locally?** Yes — this analysis was run end-to-end on a machine with 16.9 GB total / ~5.4 GB available RAM at the time, without any chunking, and completed comfortably. A standard laptop with 8GB+ RAM should handle the full dataset with sensible dtypes (categoricals for IDs, `int16`/`int32` for sales counts, `float32` for prices).
- **When chunking/efficient dtypes might still matter:** if the sales table is melted from wide (one row per store-item) to long format (one row per store-item-day, ~59M rows) for feature engineering, memory use will grow substantially (a long-format frame with several engineered float columns could reach multiple GB) — efficient dtypes and possibly chunked/incremental processing become worth planning for at that stage, though not strictly required for the raw files as they are.

---

## 17. Leakage Analysis

For each feature, asking: *"Would we actually know this value at the exact moment we make the forecast?"*

**Safe (confirmed available for the full future window, per Section 13):**
- Calendar attributes (date, weekday, month, year, `wm_yr_wk`)
- Event name/type (`event_name_1/2`, `event_type_1/2`) — future values present and non-corrupted
- SNAP flags (`snap_CA/TX/WI`) — future values present
- Price (`sell_price`) — future values present, though see caveat below
- Static identifiers (`item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`)
- Lagged/rolling versions of past sales (e.g., sales 28+ days ago) — safe as long as the lag window doesn't reach into the forecast horizon itself

**Potentially unsafe / needs care:**
- **`sell_price` as a future-known feature is technically provided by this dataset**, but modelers should be aware this is a simplification versus a live production setting where the *exact future retail price* usually isn't known this precisely this far out — using it is legitimate *for this dataset/competition* but is worth flagging explicitly as a modeling assumption, not a universal truth about retail forecasting.
- **Any feature engineered from `sales_train_validation.csv` vs `sales_train_evaluation.csv`**: because the evaluation file's extra 28 days (`d_1914`–`d_1941`) are literally the same period that the *validation*-style forecast would have targeted, using evaluation-file data as "future truth" to tune a model that will be scored against the same window would leak the target back into training. This needs a clear train/validation split decision later.
- **Non-lagged same-day aggregate features** (e.g., a feature like "total store sales that day" computed across all items) would leak information across the item panel if not carefully time-fenced — not present in the raw files, but a risk if engineered carelessly downstream.

**Definitely leaking (not present as a risk here, but worth stating for completeness):**
- Actual future sales values (`d_1942` onward) — not provided in any file, so no risk of accidentally including them; the task correctly withholds exactly the target.

No features are removed at this stage — this section only flags risk for the future feature-engineering phase, per the scope of this investigation.

---

## 18. Forecasting Feasibility

Based only on what's in the files:

- **Can we forecast 28 days?** Yes — this is exactly the horizon `sample_submission.csv` asks for (`F1`...`F28`), and 1,941 days of history (≈69× the horizon length) is ample for that horizon.
- **Can we forecast 56 days?** Nothing in the files asks for this, but from a data-sufficiency standpoint alone, 1,941 days of history could support a longer horizon — the files just don't request one.
- **Can we forecast an entire year (365 days)?** The files don't ask for this either. Purely on data volume, 5.3 years of history is enough to *attempt* year-long structure recognition, but forecast accuracy would degrade the further out you go — no year-long validation exercise was performed in this investigation, so this is a feasibility statement about data volume, not a claim about achievable accuracy.
- **How much historical data do we have?** 1,941 days ≈ 5.31 years, confirmed continuous with zero gaps.
- **Enough for weekly seasonality?** Yes, easily — over 277 full weekly cycles are present, and a clear weekend-vs-weekday pattern is already visible in the raw aggregate (Section 9).
- **Enough for yearly seasonality?** Yes, with appropriate caveats — 5+ full or near-full calendar years (2012–2015 are complete years; 2011 and 2016 are partial) give multiple year-over-year comparison points, which is meaningfully more than a "two-month" dataset would allow. This report does not claim the yearly seasonality has been rigorously measured (no decomposition was run) — only that there is enough data to do so.
- **Future covariates available:** calendar (date/weekday/month/events/SNAP) and price, confirmed present for the entire 28-day forecast window (Section 13).
- **Biggest limitations, stated honestly:**
  1. **No promotion signal at all** — a real driver of retail demand spikes is not observable in this dataset.
  2. **68% zero-inflation** makes this fundamentally an intermittent-demand problem, not a smooth continuous-forecasting problem — many standard techniques will need adaptation.
  3. **Structural "leading zero" blocks from product introduction** (Section 14) affect roughly two-thirds of series and must be handled explicitly, or they will bias any model that treats all zeros identically.
  4. **No explicit stockout/inventory signal** — a zero could mean "no demand" or "out of stock"; the files don't distinguish these, and this report did not attempt to infer the difference.
  5. Price is given as an exact future-known value, which is a simplification worth remembering when interpreting results.

---

## 19. Key Findings

1. **The dataset is the 5-file structure of a public retail-forecasting competition** (sales_train_validation/evaluation, calendar, sell_prices, sample_submission) — but every fact used in this report (row/column counts, date ranges, hierarchy sizes, missingness, price coverage) was independently re-derived from the actual files rather than assumed from that resemblance.
2. **~5.3 years of fully continuous daily history** (2011-01-29 to 2016-05-22), far beyond a "couple of months" — supports both weekly and (with appropriate caveats) yearly seasonality analysis.
3. **The 28-day forecast horizon's calendar and price features are already known** (`calendar.csv` and `sell_prices.csv` both extend 28 days past the last day of actual sales) — this is a materially favorable feature-availability situation, verified directly, not assumed.
4. **No promotion data exists anywhere in the dataset** — only raw price. This is a real limitation for demand-spike modeling.
5. **68% of all daily observations are exactly zero** — intermittent demand is the norm, not the exception, and 78% of all series have more zero-days than nonzero-days.
6. **Sales are cleanly structured and essentially defect-free**: zero missing values in sales, zero negative sales, zero negative/zero prices, zero duplicate IDs, zero date gaps.
7. **The one real data-quality nuance** is that roughly two-thirds of store-item series have a partial (not full-history) price record, which lines up with — and appears to explain — leading zero-sales blocks for products introduced mid-timeline; this needs explicit handling, not blind imputation.
8. **The forecasting unit is confirmed to be store × item** (30,490 independent series), with hierarchy metadata (item→dept→cat, store→state) available but the submission file only requiring bottom-level (store-item) forecasts.
9. **Meaningful cross-store and cross-category heterogeneity exists** — e.g., store `CA_3` generates 17% of all units sold company-wide (2.7× the lowest store), and FOODS accounts for 69% of total volume despite being 47% of the panel.

---

## 20. Important Unknowns

- Whether zero-sales days in the middle of a product's listed lifetime represent genuine no-demand vs. unrecorded stockouts — **not available in the provided dataset** (no inventory/stock column exists).
- Whether the visually-apparent price-vs-sales relationship for the sampled top item (Section 11) generalizes across the panel or is a one-off pattern — not tested at scale in this investigation.
- Whether any promotional activity happened that simply isn't recorded — **not available in the provided dataset**.
- Whether `sales_train_validation.csv` should be used at all (given it's a strict subset of `sales_train_evaluation.csv`) or is purely a legacy artifact of a two-stage competition — a usage decision, not something the data itself resolves.
- What exactly the private/final evaluation window (`d_1942`–`d_1969`) will be scored against, since sample_submission's `_evaluation` block currently has no ground truth in any file — **not available in the provided dataset**.
- Precise quantification of yearly seasonality strength, holiday-effect size, and SNAP-effect size beyond the single illustrative check in Section 11/16 — not computed in this investigation, flagged as follow-up work.

---

## 21. Recommended Next Steps

1. Decide how to treat `sales_train_validation.csv` vs `sales_train_evaluation.csv` (e.g., use evaluation as the single source of truth, since it's a strict superset) and how to structure a local train/validation split that doesn't leak the `d_1914`–`d_1941` window into training if that window will be used for validation.
2. Design explicit handling for pre-listing "leading zero" blocks (Section 14) — e.g., using the first priced week per store-item as a "series start" marker rather than treating all history uniformly.
3. Quantify holiday/event and SNAP effects more rigorously (effect sizes, per-category breakdowns) before deciding how to encode them as features.
4. Investigate whether price changes correlate with sales changes broadly (not just the single example item), to judge whether an engineered "promotion proxy" from price drops is worth building given there's no real promotion flag.
5. Decide on an approach appropriate for heavy zero-inflation/intermittent demand (a modeling decision — intentionally deferred, not made in this report).
6. Only after the above: move to feature engineering and model selection.

---

## Appendix: Files produced by this investigation (`analysis_output/`)

- `step1_inventory.json`, `step2_calendar_summary.json`, `step3_hierarchy_summary.json`, `step4_sales_stats.json`, `step5_price_summary.json`, `step6_submission_quality.json` — raw computed stats backing this report
- `series_summary.csv` — per-store-item summary stats (total/mean/std/max sales, % zero days) for all 30,490 series
- `daily_total_sales.csv`, `store_daily_sales.csv`, `cat_daily_sales.csv` — daily aggregates used for time-series charts
- `store_summary.csv`, `cat_summary.csv`, `dept_summary.csv` — hierarchy-level rollups
- `calendar_full_readable.csv` — calendar.csv with parsed dates, for reference
- `sample_price_history.csv` — example price history for two sample items
- `01_overall_sales_over_time.png` through `09_snap_effect.png` — exploratory charts referenced throughout this report

No file inside `DATASET_m5-forecasting-accuracy/` (the original dataset) was modified.
