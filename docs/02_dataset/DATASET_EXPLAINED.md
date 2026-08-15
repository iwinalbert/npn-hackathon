# Problem Statement 11 — M5 Retail Demand Forecasting Dataset Explained

### Complete Dataset Guide: Files, Columns, Relationships, Forecasting Role, and Data Availability

*A beginner-friendly study guide built from the actual dataset investigation (`DATASET_SUMMARY.md`) and the real files in `DATASET_m5-forecasting-accuracy/`. Every number below was computed from the actual files — nothing is invented, and anywhere the investigation could not confirm something, it is explicitly marked "Not available in the provided dataset."*

---

## 1. Executive Summary

**What is this dataset?**
It is retail sales data from a real chain of stores. It records, for **3,049 products** sold across **10 stores** in **3 US states**, how many units of each product were sold **every single day** for over five years. Alongside the sales, we're given a calendar (with holidays and a food-assistance-benefit flag) and the weekly selling price of every product.

**What are we trying to predict?**
For every one of the **30,490 product-store combinations**, how many units will sell on each of the **next 28 days**.

**What information do we have?**
- 1,941 days of actual historical daily sales for every product-store combination
- A calendar covering 1,969 days total — including the 28 days right after the sales history ends
- Weekly prices for every product-store combination — also covering those same 28 future days
- No promotion/discount data of any kind (confirmed absent — see Section 16)

**What is the forecasting horizon?**
28 days — from the day after the last known sale to the last day of the calendar file.

**The core numbers, all verified from the actual files:**

| Quantity | Value |
|---|---|
| Products | **3,049** |
| Stores | **10** |
| States | **3** (California, Texas, Wisconsin) |
| Store-item time series | **30,490** (= 3,049 × 10) |
| Calendar days covered | **1,969** (2011-01-29 → 2016-06-19) |
| Historical sales through | **2016-05-22** |
| Forecast window | **2016-05-23 → 2016-06-19** |
| Forecast horizon | **28 days** |

**In plain language:** this is not "one product, one time series" forecasting. It's 30,490 separate daily time series that all need forecasting at once, most of which sell only occasionally (more on that in Section 13), with a calendar and price file that conveniently already tell us what the world looks like during the 28 days we need to predict — we just don't know the sales themselves yet.

---

## 2. Big-Picture Dataset Architecture

The five files fit together like this:

```
┌───────────────────────────────┐
│  sales_train_evaluation.csv   │   ← PRIMARY sales fact table
│  30,490 rows × 1,941 days     │      (superset — includes everything
│                                │       sales_train_validation.csv has,
│  ├── item_id / dept_id/       │       plus 28 more days)
│  │   cat_id  (product info)   │
│  ├── store_id / state_id      │
│  │   (location info)          │
│  └── d_1 … d_1941              │
│      (daily unit sales)        │
└───────────────┬────────────────┘
                │ joins on "d" (day index) and (store_id, item_id)
                │
     ┌──────────┴───────────┐
     ▼                       ▼
┌─────────────────┐   ┌─────────────────────┐
│  calendar.csv    │   │  sell_prices.csv     │
│  1,969 days      │   │  6,841,121 rows      │
│  ├ date          │   │  ├ store_id          │
│  ├ weekday       │   │  ├ item_id           │
│  ├ event_name/   │   │  ├ wm_yr_wk (week)   │
│  │  type 1 & 2   │   │  └ sell_price         │
│  └ snap_CA/TX/WI │   │                       │
│                  │   │  Both files extend    │
│  Extends 28 days │   │  28 days PAST the     │
│  past the last   │   │  last known sale —    │
│  known sale.     │   │  this is the future   │
└─────────────────┘   └─────────────────────┘
                │                       │
                └───────────┬───────────┘
                             ▼
                  Historical sales + future
                  calendar + future price
                             │
                             ▼
                   28-DAY FORECAST
                 (for every store-item)
                             │
                             ▼
              ┌────────────────────────────┐
              │  sample_submission.csv      │
              │  60,980 rows × 29 columns   │
              │  id, F1, F2, … F28           │
              │  Defines the exact shape     │
              │  your predictions must take  │
              └────────────────────────────┘

┌────────────────────────────────┐
│ sales_train_validation.csv      │  A shorter, earlier snapshot of the
│ 30,490 rows × 1,913 days        │  SAME 30,490 series — identical IDs,
│ (subset of sales_train_         │  identical structure, just missing
│  evaluation.csv)                │  the last 28 days that evaluation has.
└────────────────────────────────┘
```

**In words:** `sales_train_evaluation.csv` is the main historical-sales table — one row per product-store combination, one column per day. `calendar.csv` and `sell_prices.csv` both attach extra context to those daily columns (calendar via the day index, price via the week index), and — importantly — both of them keep going for **28 days after the sales data stops**. That 28-day extension is exactly the forecast window, and it's why we already have calendar and price information for it. `sample_submission.csv` just tells us the exact output format expected. `sales_train_validation.csv` is not a separate dataset with new information — it's confirmed to be a strict subset (same IDs, same values, just 28 fewer days) of `sales_train_evaluation.csv`.

---

## 3. Dataset File #1 — `sales_train_validation.csv`

**What it contains:** daily unit sales for all 30,490 store-item combinations, from day `d_1` (2011-01-29) through day `d_1913` (2016-04-24) — 1,913 days.

**Why it exists:** it appears to be an earlier snapshot of the same underlying sales table — everything in it (every ID, every value for the days it covers) is also present, unchanged, inside `sales_train_evaluation.csv`. It was verified that both files share an identical set of 30,490 base IDs in identical order.

**What one row represents:** one specific **product, in one specific store**, tracked over time. For example, the row with `id = HOBBIES_1_001_CA_1_validation` represents item `HOBBIES_1_001` sold in store `CA_1`, and nowhere else.

**What the ID represents:** the `id` column is built as `{item_id}_{store_id}_validation`, e.g. `HOBBIES_1_001_CA_1_validation`. It's a compound key — read left to right, it tells you the product and the store in one string.

**What the `d_1, d_2, d_3, …` columns represent:** each one is a single calendar day, in sequence, starting from the very first day in the dataset. `d_1` is always 2011-01-29 for every row. The actual calendar date for `d_1913` (or any `d_n`) has to be looked up in `calendar.csv`, which is not stored in this file directly — it only stores the sequence number.

**The wide-format structure, explained simply:**

```
item + store  (one row)
    │
    ├── d_1  = units sold on day 1  (e.g. Saturday, 2011-01-29)
    ├── d_2  = units sold on day 2  (Sunday,   2011-01-30)
    ├── d_3  = units sold on day 3  (Monday,   2011-01-31)
    ...
    └── d_1913 = units sold on day 1913 (2016-04-24)
```

This is called **wide format**: instead of one row per (item, store, day) — which would be a ~58-million-row table — the file uses one row per (item, store), with **each day as its own column**. This makes the file compact and fast to scan, but it also means any date-based operation (joining to the calendar, computing a rolling average, etc.) requires first "melting" the `d_` columns back into a normal (item, store, date, sales) long table, or working with the wide array directly using date-aligned column positions.

**Item and store information:** stored directly on the row, as separate columns — `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` — so you never need a separate lookup table to know which department or state a row belongs to.

**Why this is an important historical-sales dataset:** it establishes the full 30,490-series structure and gives 1,913 days of ground truth. However, since every one of its rows and values is also contained in `sales_train_evaluation.csv` (see next section), the *evaluation* file is the more complete — and therefore preferred — source of historical sales for building anything going forward.

---

## 4. Dataset File #2 — `sales_train_evaluation.csv`

**What it contains:** the exact same 30,490 store-item structure as `sales_train_validation.csv`, but extended to day `d_1941` (2016-05-22) — **28 more days of real, actual sales** than the validation file has.

**How it differs from `sales_train_validation.csv`:**

| | `sales_train_validation.csv` | `sales_train_evaluation.csv` |
|---|---|---|
| Rows | 30,490 | 30,490 (same IDs) |
| Day columns | `d_1` … `d_1913` | `d_1` … `d_1941` |
| Last date covered | 2016-04-24 | **2016-05-22** |
| `id` suffix | `_validation` | `_evaluation` |
| Relationship | Strict subset | **Strict superset** (contains everything validation has, plus 28 more days) |

**Why the evaluation version exists — explained with a timeline:**

```
d_1 ─────────────────────────────────── d_1913 ── d_1914 ─── … ─── d_1941
2011-01-29                              2016-04-24  2016-04-25        2016-05-22
│                                            │            │                │
└── sales_train_validation.csv covers here ──┘            │                │
│                                                          │                │
└── sales_train_evaluation.csv covers all the way to here ────────────────┘
                                                            └── these 28 days
                                                                are ONLY in the
                                                                evaluation file
```

This pattern is typical of a two-stage forecasting competition: an earlier round asked competitors to predict the 28 days from `d_1914`–`d_1941` using only the `validation` file as history (i.e., those days were originally the "unknown future" for that round). At the point this dataset was assembled, those 28 days' actual results have now been revealed and folded into `sales_train_evaluation.csv`, which is why the evaluation file simply looks like the validation file with 28 extra columns of real data.

**Its role in this problem:** `sales_train_evaluation.csv` is the file that should be used as **the primary historical-sales source** for this project — it has strictly more real ground truth than the validation file, with no information lost. The genuinely unknown period we must predict is the **next** 28 days after evaluation's last day (`d_1942`–`d_1969`, i.e. 2016-05-23 → 2016-06-19) — which is not contained as real sales data in *any* file (see Section 10).

---

## 5. Dataset File #3 — `calendar.csv`

`calendar.csv` has **1,969 rows** (one per date) and **14 columns**. It's the file that turns the abstract `d_1`, `d_2`, ... day indices into real calendar information, including information about events and a food-assistance-benefit flag.

| Column | What it means | Example | Why it matters for forecasting | Available for future dates? |
|---|---|---|---|---|
| `date` | The real calendar date | `2011-01-29` | Lets you join sales day-indices to actual dates, compute date-based features | Yes — present for all 1,969 rows |
| `d` | The day-index used as the sales files' column names (`d_1`, `d_2`, …) | `d_1` | This is the join key between `calendar.csv` and the sales files | Yes — goes up to `d_1969` |
| `weekday` | Day name | `Saturday` | Retail sales are strongly weekday/weekend dependent (confirmed in the investigation — weekends are the highest-volume days) | Yes |
| `wday` | Numeric day-of-week, 1=Saturday … 7=Friday | `1` | Same as `weekday`, but numeric — easier to feed into a model | Yes |
| `month` | Calendar month | `1` | Supports monthly/seasonal features | Yes |
| `year` | Calendar year | `2011` | Supports year-over-year features | Yes |
| `wm_yr_wk` | An internal year-week code | `11101` | This is the join key to `sell_prices.csv`, since prices are recorded weekly, not daily | Yes — goes up to `11621`, covering the future window |
| `event_name_1` | Name of a special event on this date, if any | `SuperBowl` | A real demand driver for certain product categories on specific dates | Yes — non-missing for all 28 future days (verified) |
| `event_type_1` | Category of `event_name_1`: `Sporting`, `Cultural`, `National`, or `Religious` | `Sporting` | Groups events into 4 types, useful if there are too many individual event names to model separately | Yes |
| `event_name_2` | A second, simultaneous event on the same date (rare — only 5 dates in the whole file have one) | `Easter` | Handles the rare case of two events landing on the same day | Yes |
| `event_type_2` | Category of `event_name_2` | `Cultural` | Same idea as `event_type_1` | Yes |
| `snap_CA` | Whether SNAP benefits are usable that day in California (0 or 1) | `1` | A real government-benefit-driven demand signal, especially for FOODS | Yes — non-missing for all 28 future days (verified) |
| `snap_TX` | Same, for Texas | `0` | — | Yes |
| `snap_WI` | Same, for Wisconsin | `0` | — | Yes |

> **What is "SNAP"?** Per the calendar column names and the fact that the flag is state-specific and follows a recurring ~10-days-per-month pattern, `snap_CA`/`snap_TX`/`snap_WI` most plausibly represents days when SNAP (Supplemental Nutrition Assistance Program, a US food-assistance benefit) is disbursed/usable in that state. This interpretation is consistent with everything observed in the data (state-specific, food-relevant, recurring monthly pattern) but the file itself does not include a description column — **treat the exact program definition as reasonably inferred from context, not literally stated in the file, and verify against the official problem statement/documentation if precision matters.**

**The most important thing about this file: it tells us about the FUTURE.**

`calendar.csv` doesn't stop where the sales data stops. It keeps going for **28 more days** — exactly the days we need to forecast:

```
Historical sales (d_1 … d_1941)
        +
Future calendar information (d_1942 … d_1969) ← ALREADY IN calendar.csv
        ↓
   28-day forecast
```

This was directly verified: `calendar.csv` has 1,969 rows (`d_1969` is its last), while the sales files stop at `d_1941`. The 28-day gap between them is precisely the forecast window, and every calendar/event/SNAP column in that window has **zero missing values** — meaning we already know, for every one of the 28 days we must forecast, what day of the week it is, whether it's a holiday, and whether SNAP is active. This matters because it means the model isn't forecasting blind — it can "see" the calendar context of the days it's predicting for, even though it can't see the sales.

---

## 6. Dataset File #4 — `sell_prices.csv`

`sell_prices.csv` is the largest file by row count: **6,841,121 rows**, but only **4 columns**.

| Column | What it means | Example |
|---|---|---|
| `store_id` | Which store | `CA_1` |
| `item_id` | Which product | `HOBBIES_1_001` |
| `wm_yr_wk` | Which week (joins to `calendar.wm_yr_wk`) | `11325` |
| `sell_price` | The price of that item, in that store, during that week | `9.58` |

**Why is price weekly instead of daily?** This is exactly what the actual file structure shows: there is one price row per (`store_id`, `item_id`, `wm_yr_wk`) combination — there is no daily price column anywhere. In other words, the file itself was built at weekly granularity; a product's price is treated as constant for an entire week (identified by `wm_yr_wk`) and only changes at week boundaries. This is a structural fact of the provided file, not an assumption.

**Connecting price to daily sales:**

```
Store + Item + Week
        ↓
     Price   (looked up from sell_prices.csv via wm_yr_wk)
        ↓
Daily demand context
   (every day within that same week shares the same price,
    because calendar.csv maps each day to exactly one wm_yr_wk)
```

To find the price that was in effect on any specific day, you look up that day's `wm_yr_wk` in `calendar.csv`, then find the matching (`store_id`, `item_id`, `wm_yr_wk`) row in `sell_prices.csv`.

**Future prices are available — verified.** `sell_prices.csv`'s `wm_yr_wk` values go up to **11621**, which is exactly `calendar.csv`'s maximum `wm_yr_wk` (also 11621, the week containing the very last calendar day, `d_1969`). The last day of real sales (`d_1941`) falls in week 11617. That means price data extends a further 4 weeks (28 days) past the last known sale — **covering the entire forecast window**. So, just like the calendar, we already know what price each product will be sold at during the 28 days we need to forecast.

**The partial price-history issue (found during the investigation):** not every store-item pair has a price for every one of the 282 distinct weeks in the file. The number of priced weeks per store-item pair ranges from **19 weeks up to the full 282 weeks** (only 35.9% of pairs — 10,932 of 30,490 — have a price for literally every week). This was cross-checked directly against sales: for an example pair with a short price history (item `FOODS_3_595` in store `CA_1`, whose price history starts at week `11603`), sales were **exactly zero for every single day before that first priced week (1,841 consecutive zero days), and then 126 total units afterward.** This strongly indicates the product simply wasn't being sold in that store yet — not that demand was zero. This "leading zero" pattern is discussed further in Section 14.

---

## 7. Dataset File #5 — `sample_submission.csv`

**Why it exists:** it's a template that shows exactly what shape a set of predictions must take before it can be evaluated or submitted. It contains no useful sales information itself — every forecast value in it is currently a placeholder `0`.

**What its rows represent:** one row per store-item combination, just like the sales files — but there are **60,980 rows**, exactly double the 30,490 store-item combinations, because the file has **two blocks**: one set of 30,490 rows with `id` ending in `_validation`, and another 30,490 rows ending in `_evaluation`. Both sets of IDs, once the suffix is removed, match the exact same 30,490 store-item universe as the sales files (verified).

**What the `F1`–`F28` columns mean:** `F1` is "the forecast for day 1 of the horizon," `F2` is "day 2," and so on through `F28` — the 28-day-ahead forecast values, one column per day, in order.

**Simple example of the required shape:**

| id | F1 | F2 | F3 | ... | F28 |
|---|---|---|---|---|---|
| `HOBBIES_1_001_CA_1_validation` | 0 | 0 | 0 | ... | 0 |
| `HOBBIES_1_001_CA_1_evaluation` | 0 | 0 | 0 | ... | 0 |
| `HOBBIES_1_002_CA_1_validation` | 0 | 0 | 0 | ... | 0 |
| ... | | | | | |

**How this connects to modeling later:** whatever forecasting approach is eventually used, its final output must be reshaped into exactly this structure — one row per store-item (times two, for the `_validation`/`_evaluation` blocks), 28 forecasted values per row, matching column names `F1`...`F28`. **No prediction is being built in this document** — this section only explains the target output format so it's understood ahead of time.

---

## 8. How the Five Files Work Together

A concrete, step-by-step walkthrough of how you'd answer "what do we know about item `HOBBIES_1_001` in store `CA_1` on a given day, and what do we know about the day we need to forecast?"

**Step 1 — Find the item/store combination in the sales data.**
Look up `id = HOBBIES_1_001_CA_1_evaluation` in `sales_train_evaluation.csv`. This row has 1,941 daily sales values (`d_1`...`d_1941`), plus its `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`.

**Step 2 — Find the corresponding date for any given day index.**
Say we care about `d_500`. Look up `d = "d_500"` in `calendar.csv` to get the real `date`, `weekday`, `month`, `year`, and `wm_yr_wk` for that day.

**Step 3 — Look up the calendar information for that date.**
From that same `calendar.csv` row: was there an `event_name_1`? Was `snap_CA` active (since this row's store is in California)?

**Step 4 — Look up the corresponding weekly price.**
Take the `wm_yr_wk` found in Step 2, and find the row in `sell_prices.csv` where `store_id = CA_1`, `item_id = HOBBIES_1_001`, `wm_yr_wk` matches. That gives the price in effect on `d_500`.

**Step 5 — Use all available information to reason about demand.**
Now you have, for `d_500`: the actual sales value, the day of week, whether it was a holiday or SNAP day, and the price — a complete daily picture for that single item-store-day.

**For the forecast window (`d_1942`–`d_1969`), the same steps 2–4 all still work** — because `calendar.csv` and `sell_prices.csv` both already contain rows for those days — but **Step 1 has no sales value to read**, because that's exactly what needs to be predicted.

---

## 9. Understanding the 30,490 Time Series

**3,049 items × 10 stores = 30,490 series.**

This means the same product is tracked **separately in every store it's sold in**:

```
Store CA_1  +  Item HOBBIES_1_001   →  one time series (1,941 daily values)
Store CA_1  +  Item HOBBIES_1_002   →  a different time series
Store CA_2  +  Item HOBBIES_1_001   →  yet another time series (same item, different store)
Store TX_1  +  Item FOODS_3_090     →  another time series
   ...                                     ...  (30,490 total)
```

Each of these 30,490 combinations has **its own history, its own typical volume, and its own pattern of zero/nonzero days** — verified in the investigation to differ meaningfully by store (e.g., store `CA_3` alone generates 17% of total company-wide units sold, over 2.7× the lowest-volume store) and by item/category (FOODS items sell far more often than HOUSEHOLD or HOBBIES items).

**Why this makes the problem much harder than forecasting one product:**
- A method that works well for a fast-moving item like `FOODS_3_090` in its best store (over 130 units/day on average) may not work at all for a slow-moving item that sells a total of 15 units across 5+ years.
- Any forecasting approach has to work — or be adapted to work — across this entire range of behavior, 30,490 times over, not just once.
- Aggregate patterns (like the overall upward trend or weekend effect discussed in Section 12) may not hold uniformly for every individual series.

---

## 10. Date and Forecasting Timeline

All dates below come directly from `calendar.csv` and the day-column counts of the sales files.

```
   2011-01-29 ──────────────────────────────────────────► 2016-05-22
        │                  HISTORICAL SALES                    │
        │         (1,941 continuous days, no gaps —            │
        │          this is what the model can learn from)      │
        └────────────────────────────────────────────────────┬─┘
                                                               │
                                          2016-05-23 ──────────┼────────► 2016-06-19
                                               │        28-DAY FORECAST          │
                                               │   (what the model must predict) │
                                               └──────────────────────────────────┘
```

**What is historical:** 2011-01-29 through 2016-05-22 — 1,941 days of real, actual sales values, present in `sales_train_evaluation.csv`.

**What is future:** 2016-05-23 through 2016-06-19 — 28 days. No sales values exist for this window in any file.

**What the model knows about the future window:** the calendar (weekday, month, year, events, SNAP) and the price of every product in every store — both confirmed present for all 28 of these days (Sections 5 and 6).

**What the model must predict:** the actual unit-sales values for those same 28 days, for all 30,490 store-item series — this is precisely the `F1`...`F28` structure defined by `sample_submission.csv`.

---

## 11. The Future-Information Question

A natural question when first looking at any forecasting dataset:

> **"If we only know historical sales, how can the model know about something happening in the future?"**

**With historical sales alone, the answer is: it can't.** A model trained only on past `d_1`...`d_1941` sales values has no way of knowing, on its own, that (for example) June 19, 2016 is a particular day of the week, or that a SNAP benefit period is active that day, or what price the product will be sold at.

**But this dataset gives us more than just historical sales:**

```
Historical sales
        +
Future calendar information   (already in calendar.csv, for all 28 forecast days)
        +
Future price information      (already in sell_prices.csv, for all 28 forecast days)
        ↓
A more informed forecast — the model can be given the exact weekday,
holiday/event status, SNAP status, and price for each of the 28 days
it needs to predict, even though it never sees the actual future sales.
```

**Exactly what our dataset provides, stated plainly:**
- **YES** — weekday, month, year for the forecast window (from `calendar.csv`)
- **YES** — holiday/event names and types for the forecast window (from `calendar.csv`, verified non-missing)
- **YES** — SNAP flags for the forecast window (from `calendar.csv`, verified non-missing)
- **YES** — price for the forecast window (from `sell_prices.csv`, verified via matching `wm_yr_wk` coverage)
- **NO** — actual future sales (this is the target being predicted, and is correctly withheld everywhere)

**Explicitly stated: there is NO promotion dataset in the provided files.** A complete column-by-column scan of all five files found no promotion, discount, markdown, or "on-deal" flag anywhere. This means the model has no direct, labeled way to know "this product was on promotion that day" — a real driver of retail demand spikes that this dataset simply does not expose. Any such signal would have to be *indirectly inferred* from price changes (see Section 14), which is a much weaker and noisier substitute than a true promotion flag.

---

## 12. Data Characteristics We Discovered

Straight from the investigation (`DATASET_SUMMARY.md`), with a plain-language explanation of what each one actually means for us:

| Finding | What it actually means for us |
|---|---|
| **5.3 years of history** (1,941 days) | Enough data to learn weekly patterns confidently, and enough calendar years (2011–2016) to attempt yearly-seasonality analysis later — this is not a "too little data" situation. |
| **1,969 continuous calendar days, zero gaps** | We don't need to worry about missing/interpolated dates when building date-based features — every single day in the range is accounted for. |
| **30,490 time series** | The problem is a large panel of series, not one series — any approach needs to scale across all 30,490 combinations (see Section 9). |
| **~68% zero-sales observations** | Most day-item-store combinations record no sale at all. This is the single most defining characteristic of the dataset (full discussion in Section 13). |
| **Strong intermittent-demand pattern** | Confirms the above isn't noise — 78% of all 30,490 series have *more* zero-days than nonzero-days across their whole history. |
| **Partial price histories for ~64% of series** | Roughly two-thirds of store-item pairs don't have a price recorded for the entire timeline — their price record starts partway through, lining up with the product being introduced to that store later (Section 14). |
| **No promotion data** | Confirmed absent from all five files (Section 11) — a real limitation, not an oversight in this investigation. |
| **No negative sales values** | The 40,241,819 zero observations are genuine "nothing sold," not corrupted or clipped negative values — the data is trustworthy here. |
| **No duplicate IDs** | Every store-item row in the sales files, and every (store, item, week) row in the price file, is unique — no accidental double-counting risk from duplicated rows. |
| **No missing sales values** | All 59,181,090 daily sales cells in `sales_train_evaluation.csv` are non-null integers — there's no need to decide how to impute missing sales, because there aren't any. |

This section is a summary of *what was found* — it deliberately does not recommend any particular modeling technique.

---

## 13. Zero Sales / Intermittent Demand

This is the most important characteristic of the dataset to understand before modeling, so it gets its own section.

**A simple example — a week of sales for one product in one store:**

```
Day 1 → 0
Day 2 → 0
Day 3 → 0
Day 4 → 7
Day 5 → 0
Day 6 → 0
Day 7 → 12
```

Notice: most days show **zero** units sold, and then, seemingly out of nowhere, a day shows a real sale (7 units, then later 12 units). This pattern — long stretches of zero punctuated by occasional nonzero sales — is called **intermittent demand**.

**Why this is called intermittent demand:** the word "intermittent" means "occurring at irregular intervals, not continuously." Unlike, say, daily electricity usage (which is nonzero essentially every day, just varying in amount), many retail products are only purchased by a customer on *some* days — especially slower-moving items in smaller stores. On the days nobody buys the product, the true demand might still be "1 person might have bought it if they'd been in the store," but the recorded value is 0 because no transaction happened.

**Why this matters for our dataset specifically, using the real numbers found:**
- **68% of all 59.18 million daily observations are exactly 0.**
- The median daily sales value, across the *entire* dataset, is **0** — meaning more than half of all observations are zero.
- **Not a single one of the 30,490 series is "always selling"** — every single store-item pair has at least one zero-sales day somewhere in its 1,941-day history.
- At the same time, **not a single series is "always zero"** either — every series has some nonzero sales at some point, so there's always a genuine signal to try to forecast, it's just sparse for many series.

**Why this is an important characteristic of our dataset:** it means the "typical" retail demand pattern here is closer to "occasional purchases" than "steady daily flow." A forecasting approach that assumes smooth, continuously-varying demand (the way, for example, temperature or stock prices move) will be a poor fit for a large share of these series. This section deliberately does not recommend a specific modeling technique — that decision comes later — but understanding this pattern now is essential context for whatever comes next.

Chart reference: `analysis_output/03_zero_sales_distribution.png` (shows how zero-sales percentage is distributed across all 30,490 series) and `analysis_output/06_example_series.png` (shows three real example series — high-volume, typical, and heavily intermittent).

---

## 14. Price Data Limitation

The investigation found that **19,558 of the 30,490 store-item pairs (64.1%) do not have a price recorded for every week in the dataset** — their price history starts later than the very first week.

**Why "missing price history" does NOT automatically mean "missing sales":** it would be easy to assume a gap in price data is a data-quality problem (like a missing value that needs to be filled in). But when this was checked against actual sales for an example case — item `FOODS_3_595` in store `CA_1`, whose price record starts at week `11603` — the sales record showed **exactly 0 units sold for all 1,841 days before that first priced week**, and then real, nonzero sales (126 units total) after it. The zeros before the first price aren't missing data or errors — they're 100% consistent with the product simply not being available for sale in that store yet.

**What this suggests:** the investigation's working explanation — based directly on this cross-check, not general assumption — is that **many products entered a given store's assortment partway through the 5.3-year timeline**, rather than being available from day one everywhere. A store-item pair with only 19 weeks of price history most likely represents a product introduced quite recently (relative to the full dataset), not a product with unreliable price records.

**Why this matters going forward:** this "leading zero" block needs to be treated differently from an *ordinary* intermittent-demand zero (Section 13). A zero recorded *after* a product was already listed for sale reflects genuine day-to-day demand variability. A zero recorded *before* a product's first price entry likely reflects "not for sale yet," which is a different situation entirely and could bias a naive model if the two are not distinguished. This is flagged here as something to handle explicitly later — no fix has been applied to the data at this stage.

---

## 15. What Information Is Available at Forecast Time?

| Information | Available historically? | Available for the forecast period (2016-05-23 → 2016-06-19)? | Useful context | Notes |
|---|---|---|---|---|
| Historical sales | Yes (1,941 days) | No — this is the target | — | The exact thing being predicted |
| Calendar (date/weekday/month/year) | Yes | **YES** | High | Verified present for all 28 future days |
| Holidays / events | Yes | **YES** | Medium (sparse — only 8.2% of all days have any event) | `event_name_1/2`, `event_type_1/2`, non-missing for future window |
| Weekday | Yes | **YES** | High | Strong weekly pattern confirmed in the data |
| SNAP flag | Yes | **YES** | Medium–High for FOODS | Verified non-missing for future window, differs by state |
| Price | Yes | **YES** | High | Verified via matching `wm_yr_wk` coverage into the future window |
| Promotions | **NO** (not in the dataset at all) | **NO** | — | Not available in the provided dataset — no such column exists in any file |
| Store | Yes (static) | Yes (static) | High | `store_id`, `state_id` |
| Product | Yes (static) | Yes (static) | High | `item_id` |
| Category / Department | Yes (static) | Yes (static) | Medium | `cat_id`, `dept_id` |

**Highlighted, as requested:**
- **YES — Future calendar information** is available.
- **YES — Future price information** is available.
- **NO — Promotion data** is available.

---

## 16. What We Are Not Given

**Important Missing Information:**

- **No promotion/discount dataset or flag anywhere in the five files.** Confirmed by a full column-by-column scan. This limits the model's ability to explain sudden demand spikes that are commonly driven by promotions in real retail settings — any such effect can, at best, be weakly and indirectly approximated from price changes, not read from a genuine label.
- **No inventory or stockout information.** There is no column indicating whether a product was actually in stock on a given day. This means a recorded `0` in the sales data could mean "no one wanted to buy it" or "it wasn't available to buy" — the dataset cannot distinguish between these two very different situations.
- **No explicit "date item was first listed" column.** The investigation *inferred* likely listing timing from the first available price week (Section 14), but there is no dedicated column that states this directly — it's a derived observation, not a labeled fact.
- **No revenue/dollar-sales column.** Only unit counts are given in the sales files; converting to revenue would require multiplying by the matching week's price yourself — this has not been done anywhere in the provided files.
- **No ground truth for the true forecast window (`d_1942`–`d_1969`).** No file contains actual sales values for this period — which is expected and correct, since that's exactly what needs to be predicted, but it's worth stating plainly: there is nothing to "peek at."

Each of these gaps limits what any model — however sophisticated — can directly learn from labeled information; some of them (promotions, stockouts) may only ever be approximated indirectly, if at all.

---

## 17. Dataset → Forecasting Problem

Putting the whole picture together:

```
   Historical Sales   (1,941 days, per store-item)
          +
   Store              (10 stores, 3 states)
          +
   Item               (3,049 products, 7 departments, 3 categories)
          +
   Calendar           (weekday, month, year — known for the future too)
          +
   Events             (holidays, SNAP — known for the future too)
          +
   Price              (weekly, known for the future too)
          ↓
   ═══════════════════════════
       DEMAND FORECAST
   ═══════════════════════════
          ↓
   Next 28 Days
   (2016-05-23 → 2016-06-19)
          ↓
   30,490 store-item series
   (must match the exact shape of sample_submission.csv)
```

**In plain English:** we're combining over five years of what actually happened (sales, broken down by exact product and exact store) with everything we already know is going to be true during the next 28 days (what day of the week it'll be, whether there's a holiday, whether SNAP is active, and what the price will be) to estimate what hasn't happened yet — how many units of each of 30,490 product-store combinations will sell, one day at a time, for the next 28 days. No modeling has been performed at this stage — this section is only meant to show how the pieces of the dataset connect to the forecasting task ahead.

---

## 18. Glossary

- **Store-item series** — the sales history of one specific product in one specific store, tracked over time. There are 30,490 of these in this dataset.
- **Time series** — a sequence of values recorded in order over time (here, one value per day).
- **Forecast horizon** — how many time steps into the future you're predicting. Here: 28 days.
- **Demand** — how much of a product customers wanted to buy; approximated in this dataset by recorded unit sales.
- **Intermittent demand** — a demand pattern with many zero-value periods interrupted by occasional nonzero sales (Section 13).
- **Calendar feature** — any piece of information derived from the date itself (weekday, month, holiday, etc.) used to help a model understand time-based patterns.
- **Covariate** — a variable other than the target itself that may help explain or predict it (e.g., price, weekday, SNAP status are all covariates for sales).
- **SNAP** — in this dataset, a per-state daily flag (`snap_CA`/`snap_TX`/`snap_WI`) that is plausibly tied to the Supplemental Nutrition Assistance Program benefit-usable days; the exact program definition is inferred from context and column naming, not explicitly documented in the files themselves (see the callout in Section 5).
- **Event** — a specific named occurrence on a calendar date (e.g. "SuperBowl," "LaborDay") that might influence shopping behavior, recorded in `event_name_1`/`event_name_2`.
- **Hierarchy** — the nested grouping structure of products (item → department → category) and locations (store → state) present in this dataset.
- **Store** — one of the 10 physical retail locations in the dataset.
- **Item** — one of the 3,049 unique products in the dataset (also called `item_id`).
- **Department** — a mid-level grouping of items (7 total, e.g. `HOBBIES_1`); several departments make up a category.
- **Category** — the top-level product grouping (3 total: `FOODS`, `HOBBIES`, `HOUSEHOLD`).
- **Validation (file)** — here, `sales_train_validation.csv`, the earlier/shorter snapshot of the sales data (through 2016-04-24).
- **Evaluation (file/period)** — here, `sales_train_evaluation.csv`, the fuller snapshot of the sales data (through 2016-05-22); also used as a suffix in `sample_submission.csv` to denote the true, currently-unknown 28-day forecast block.

---

## 19. Final "Know This Before Modeling" Page

**DATASET:** 5.3 years of daily history
**PRODUCTS:** 3,049
**STORES:** 10
**STORE-ITEM SERIES:** 30,490
**HISTORICAL SALES:** Through 2016-05-22
**FORECAST WINDOW:** 2016-05-23 → 2016-06-19
**HORIZON:** 28 days
**FUTURE CALENDAR:** YES
**FUTURE PRICES:** YES
**PROMOTION DATA:** NO
**ZERO SALES:** ~68% of all observations
**MAIN DATA CHARACTERISTIC:** Intermittent demand

**Before choosing a model, we still need to determine:**

1. How to treat `sales_train_validation.csv` vs. `sales_train_evaluation.csv` — and how to build a local validation split that doesn't leak the already-revealed `d_1914`–`d_1941` window into training.
2. How to explicitly handle the "leading zero" blocks caused by products being introduced to a store partway through the timeline (Section 14), so they aren't confused with ordinary intermittent-demand zeros.
3. How strong the holiday/event and SNAP effects actually are, quantitatively — only a single illustrative check (SNAP vs. FOODS sales in California) has been done so far.
4. Whether price changes are a usable proxy signal for "something promotional happened," given there is no real promotion flag to rely on.
5. What approach is appropriate for a dataset this heavily zero-inflated — an open modeling question, intentionally not answered in this document.

*This document does not build, train, or select a forecasting model. It exists solely to make sure the dataset itself is fully understood before that work begins.*
