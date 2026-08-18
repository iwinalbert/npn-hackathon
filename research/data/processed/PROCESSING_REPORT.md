# M5 Sales Data — Processing Report

**Project:** 28-day sales forecasting hackathon
**Scope of this stage:** Data cleaning and structural preparation only — no modeling, no feature engineering, no novelty decisions.

---

## 1. Original Dataset Location

The raw M5 competition files were originally located at:

```
NPN_HACKATHON/DATASET_m5-forecasting-accuracy/
```

This folder was renamed (contents untouched) to:

```
NPN_HACKATHON/raw_dataset/
```

Files inside:

| File | Size |
|---|---|
| calendar.csv | 103,469 bytes |
| sales_train_validation.csv | 120,007,726 bytes |
| sales_train_evaluation.csv | 121,736,518 bytes |
| sell_prices.csv | 203,395,785 bytes |
| sample_submission.csv | 5,228,786 bytes |

## 2. Raw Dataset Preservation

- The folder rename was the **only** operation ever applied to the original files — no file inside was opened in write mode.
- MD5 checksums were captured **before** the rename and re-verified **after** all processing completed. All 5 files are byte-identical:

| File | MD5 | Match |
|---|---|---|
| calendar.csv | `3ffeab2991b0c8e861d008b39ea4c95c` | ✅ |
| sales_train_evaluation.csv | `b806dfc9f30a745102b708c09951f6aa` | ✅ |
| sales_train_validation.csv | `26a366a25beb57b0a8f4c7b148758f94` | ✅ |
| sample_submission.csv | `c281a69d7c011274899d92020a66e25b` | ✅ |
| sell_prices.csv | `08c591caa99e55daf3e0ccac913f7c85` | ✅ |

`raw_dataset/` remains untouched and can be treated as the ground-truth source at any future stage.

## 3. Files Processed

| File | Role |
|---|---|
| `sales_train_evaluation.csv` | **Source of truth for sales history.** Contains the full actual observed sales record, `d_1`–`d_1941`. |
| `sales_train_validation.csv` | Inspected and cross-checked, but **not** used to build the processed table. It is a truncated subset of the same data (`d_1`–`d_1913`). The quality audit confirmed its values are identical to `sales_train_evaluation.csv` on every shared day, so it carries no additional information. |
| `calendar.csv` | Joined in to map `d_*` day identifiers to real calendar dates, weekday/month/year, event flags, and SNAP flags. |
| `sell_prices.csv` | Joined in to attach the weekly selling price for each store–item combination. |
| `sample_submission.csv` | Inspected only (format check), not used in processing — it belongs to the future prediction stage. |

**Why `sales_train_evaluation.csv` and not `sales_train_validation.csv`:** the evaluation file's extra 28 days (`d_1914`–`d_1941`) are already-observed historical sales, not forecasts. Using it gives the longest real history without introducing any future/unobserved data.

## 4. Data-Quality Checks Performed

A full audit script was run directly against the raw CSVs (see `processed_dataset/_audit/quality_audit.json` for raw output). Checks covered:

**Sales (`sales_train_evaluation.csv`):**
- Missing values in ID columns and day columns
- Duplicate `id` rows
- Negative or non-numeric sales values
- Count of items, stores, departments, categories, states
- Count of item–store combinations
- Number of day columns and their range

**Calendar (`calendar.csv`):**
- Missing / duplicate dates
- Duplicate `d` identifiers
- Date-range continuity (no gaps)
- `event_name_*` / `event_type_*` consistency (a name should never appear without a type or vice versa)
- SNAP column value validity (must be 0/1)

**Sell prices (`sell_prices.csv`):**
- Missing prices
- Negative or zero prices
- Duplicate `store_id` + `item_id` + `wm_yr_wk` combinations
- Price range sanity

**Cross-table relationships:**
- Every `d_*` column used in sales exists in `calendar.d`
- `store_id` sets match between sales and prices
- `item_id` sets match between sales and prices
- Every `wm_yr_wk` referenced by the sales date range is covered by at least one price row

## 5. Problems Found

**None.** The dataset is clean:

| Check | Result |
|---|---|
| Missing sales values | 0 |
| Negative sales values | 0 |
| Duplicate sales `id` rows | 0 |
| Missing calendar dates | 0 |
| Duplicate calendar dates | 0 |
| Duplicate `d` identifiers | 0 |
| Date-range gaps | 0 |
| `event_name`/`event_type` mismatches | 0 |
| Missing prices in `sell_prices.csv` | 0 |
| Negative/zero prices | 0 |
| Duplicate store-item-week price rows | 0 |
| `d_*` columns absent from calendar | 0 |
| Store ID mismatch (sales vs. prices) | 0 |
| Item ID mismatch (sales vs. prices) | 0 |

This matches the earlier exploratory findings already in `analysis_output/` (`step1_inventory.json` through `step6_submission_quality.json`).

**One expected, non-error condition:** after joining `sell_prices` onto the long-format sales table, **12,299,413 rows (20.78%)** have no matching price. This is not missing data in the source file — it reflects that products enter a store's assortment at different points in time, so no price record exists for a store-item before that item was introduced. See Section 8.

## 6. Problems Actually Fixed

None. Per instructions, nothing was "fixed" for the sake of cleaning since no genuine defects were found.

## 7. Problems Intentionally Left Unchanged

- **Zero-sales observations (40,241,819 rows, 68.0% of all rows)** — preserved exactly as recorded. See Section 4 note below.
- **High and low sales values (spikes)**, up to a maximum of 763 units in a single day — preserved exactly as recorded.
- **Rows with no matching sell price (20.78%)** — left as `NaN`, not imputed or dropped.
- **`sales_train_validation.csv`** — left untouched in `raw_dataset/`, not merged into the processed output (see Section 3).

> **Zero-sales observations were preserved because they are valid observations and their underlying cause cannot be determined directly from the available dataset.**

> **Sales spikes were preserved because they may represent genuine demand behavior.**

No `stockout` column and no `promotion` column were created, since the raw dataset provides no direct evidence for either.

## 8. How Sales Data Was Transformed

The raw sales table is wide format — one row per item-store series, one column per day:

```
id | item_id | dept_id | cat_id | store_id | state_id | d_1 | d_2 | ... | d_1941
```

It was melted into long format — one row per (series, day):

```
id | item_id | dept_id | cat_id | store_id | state_id | d | sales
```

**Before (wide, excerpt):**

| id | item_id | store_id | d_1 | d_2 | d_3 |
|---|---|---|---|---|---|
| HOBBIES_1_001_CA_1_evaluation | HOBBIES_1_001 | CA_1 | 0 | 0 | 0 |

**After (long, excerpt):**

| id | item_id | store_id | d | sales |
|---|---|---|---|---|
| HOBBIES_1_001_CA_1_evaluation | HOBBIES_1_001 | CA_1 | d_1 | 0 |
| HOBBIES_1_001_CA_1_evaluation | HOBBIES_1_001 | CA_1 | d_2 | 0 |
| HOBBIES_1_001_CA_1_evaluation | HOBBIES_1_001 | CA_1 | d_3 | 0 |

30,490 series × 1,941 days = **59,181,090 rows**, verified against the raw file (see Section 11). No value was altered — the sum, min, max, and zero-count of `sales` in the processed table are identical to the raw file.

The original wide-format CSV was never overwritten; it remains untouched in `raw_dataset/`.

## 9. How Calendar Data Was Joined

Each row's `d` identifier was joined against `calendar.csv` on the `d` column (left join, one-to-one — `d` is unique in the calendar), adding:

`date, wm_yr_wk, weekday, wday, month, year, event_name_1, event_type_1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI`

Verification performed:
- Row count unchanged after the join (59,181,090 → 59,181,090), confirming no fan-out from duplicate `d` values.
- Every row received a non-null `date` (0 missing).
- Spot-checked `d_1`, `d_500`, `d_1000`, `d_1500`, `d_1941` against `calendar.csv` directly — all dates matched exactly (`d_1` → 2011-01-29, `d_1941` → 2016-05-22).

Calendar fields for dates beyond the sales history are not included, since the processed table only contains rows for days that actually have a sales observation (`d_1`–`d_1941`). No future sales values were introduced.

## 10. How Price Data Was Joined

Each row was joined against `sell_prices.csv` on `store_id` + `item_id` + `wm_yr_wk` (left join), adding `sell_price`.

- Row count unchanged after the join (59,181,090 → 59,181,090), confirming the join key (`store_id`, `item_id`, `wm_yr_wk`) is unique in `sell_prices.csv` (verified: 0 duplicates) so no rows were duplicated.
- **12,299,413 rows (20.78%) have `sell_price = NaN`.** This was **not** filled in. A missing price most often means the item was not yet part of that store's assortment during that week — the dataset gives no basis to invent a price for a period before a product existed on shelves. Analysts using this table in later stages should be aware that `sell_price` is nullable and decide deliberately how to treat it (e.g., filter, flag, or leave as missing) as part of feature engineering — that decision was intentionally left out of this stage.
- 5 random priced rows were independently spot-checked against `sell_prices.csv` and matched exactly.

## 11. Final Processed Dataset Structure

**File:** `processed_dataset/sales_long_full.parquet`

| Column | Type | Meaning |
|---|---|---|
| `id` | category | Original M5 series identifier (item + store + `_evaluation` suffix) |
| `item_id` | category | Product identifier |
| `dept_id` | category | Department identifier |
| `cat_id` | category | Category identifier |
| `store_id` | category | Store identifier |
| `state_id` | category | State identifier |
| `d` | string | Original day identifier (`d_1` … `d_1941`) |
| `date` | datetime64 | Calendar date for that day |
| `wm_yr_wk` | int32 | Walmart internal year-week identifier |
| `weekday` | category | Day-of-week name |
| `wday` | int8 | Day-of-week number |
| `month` | int8 | Calendar month |
| `year` | int16 | Calendar year |
| `event_name_1` / `event_type_1` | category | Primary calendar event (nullable) |
| `event_name_2` / `event_type_2` | category | Secondary calendar event (nullable) |
| `snap_CA` / `snap_TX` / `snap_WI` | int8 | SNAP benefit day flag (0/1) per state |
| `sell_price` | float32 | Weekly selling price for that store-item (nullable — see Section 10) |
| `sales` | uint16 | Observed units sold that day (raw, unaltered) |

**Shape:** 59,181,090 rows × 22 columns.

## 12. Storage Format

**Parquet** (columnar, `pyarrow` engine, `snappy` compression) was used instead of CSV because the long-format table has ~59.2 million rows; an equivalent CSV would be several gigabytes and far slower to read repeatedly during future EDA/modeling work.

- Output file: `processed_dataset/sales_long_full.parquet`
- Size on disk: **287.09 MB**
- For comparison, the three raw source files (calendar + sales_train_evaluation + sell_prices) total roughly 325 MB combined, even though the processed file also carries all the joined calendar and price columns — the reduction comes from categorical dictionary encoding and columnar compression.

## 13. Row/Column Counts

| Stage | Rows | Columns |
|---|---|---|
| Raw sales (wide) | 30,490 | 1,947 (6 ID + 1,941 day columns) |
| Long format (post-melt) | 59,181,090 | 8 |
| + calendar join | 59,181,090 | 21 |
| + price join (final) | 59,181,090 | 22 |

Row count was identical after each join, confirming no duplication was introduced.

## 14. Validation Results

Independent validation re-read the raw CSVs from scratch (not reusing the build script's own numbers) and checked:

| Check | Result |
|---|---|
| Row count = 30,490 × 1,941 | ✅ 59,181,090 |
| `item_id` set matches raw | ✅ |
| `store_id` set matches raw | ✅ |
| `id` set matches raw | ✅ |
| Sum of all sales matches raw | ✅ 66,927,173 = 66,927,173 |
| Zero-sales count matches raw | ✅ 40,241,819 |
| Max sales value matches raw | ✅ 763 |
| Min sales value matches raw | ✅ 0 |
| 5 random (id, day) sales values spot-checked | ✅ all matched |
| 5 calendar dates spot-checked (`d_1`…`d_1941`) | ✅ all matched |
| 5 random prices spot-checked against `sell_prices.csv` | ✅ all matched |
| Duplicate (`id`, `d`) pairs in processed table | ✅ 0 |
| Max day present is `d_1941` (no future days) | ✅ |
| Raw file checksums unchanged | ✅ all 5 files identical |

Full machine-readable output: `processed_dataset/_audit/validation_results.json` and `processed_dataset/_audit/quality_audit.json`.

## 15. Important Limitations

- `sell_price` is missing for 20.78% of rows because products enter store assortments at different times; this is not a data-entry error and was not imputed.
- No stockout, promotion, or event-impact labels exist — the raw dataset does not provide direct evidence for these, so none were invented.
- No forecasting features (lags, rolling statistics, days-since-last-sale, price-change flags, etc.) were created — that is explicitly deferred to a future feature-engineering stage.
- `sales_train_validation.csv` was audited but not merged into the processed table, since it is a strict subset of `sales_train_evaluation.csv`.
- The processed table covers only observed history (`d_1`–`d_1941`); it does not extend into the 28-day horizon that will need to be forecast.

---

*Generated as part of the dataset preparation stage. No model training, feature engineering, or novelty decisions were made in this stage.*
