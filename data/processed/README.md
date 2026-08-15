# processed_dataset/ — README

This folder contains two different things. **They are not interchangeable.** Read this before using either file.

## 1. `sales_long_full.parquet` — the complete processed dataset

- **59,181,090 rows × 22 columns.**
- This is the full long-format sales table, joined with calendar and price information, described in `PROCESSING_REPORT.md` / `PROCESSING_REPORT.pdf`.
- It is the **only** dataset that should be used for exploratory analysis, feature engineering, and model training.
- It has not been sampled, filtered, or altered in any way — every observation from the raw source is present.

## 2. `sales_sample_50000.csv` — human-inspection sample ONLY

- **50,000 rows × 22 columns** — the same 22 columns as the Parquet file, values copied exactly, no additional cleaning or transformation applied.
- Purpose: let a person open the file in Excel/Sheets/a text editor and visually sanity-check the data without loading 59 million rows.
- **This is not a random slice of the first 50,000 rows.** It is a stratified random sample (fixed seed = 42, reproducible) drawn proportionally across:
  - all 10 stores
  - six sales-magnitude buckets (0, 1–2, 3–5, 6–15, 16–50, 51+) — so zero-sales rows *and* spikes are both represented
  - rows with a matched price vs. rows with a missing price
  - event days vs. ordinary days
- Because the strata are sampled proportionally, the sample's overall statistics closely track the full dataset's statistics (see table below) — it is representative, not just varied.
- Sorted by `date`, then `store_id`, then `item_id` for readability.

### Sample coverage vs. full dataset

| Metric | Full dataset | Sample (50,000 rows) |
|---|---|---|
| Unique dates | 1,941 | 1,941 (all present) |
| Unique stores | 10 | 10 (all present) |
| Unique items | 3,049 | 3,049 (all present) |
| % zero-sales rows | 68.00% | 68.00% |
| % rows missing a price | 20.78% | 20.79% |
| % rows on an event day | 8.14% | 8.14% |

## Rules for using these files

- **DO NOT** treat `sales_sample_50000.csv` as the complete dataset. It intentionally omits 99.92% of the rows and is meant only for visual spot-checking.
- **DO NOT** run analysis, feature engineering, or model training against the CSV sample — results from it would not be representative of the true series lengths per item/store, which matter for time-series forecasting.
- **DO** use `sales_long_full.parquet` for all future ML work (EDA, feature engineering, training, evaluation).
- Both files were derived read-only from `sales_long_full.parquet`; neither the Parquet file nor `raw_dataset/` was modified while creating the sample (checksums verified — see `_audit/sample_validation_results.json`).

## Other files in this folder

- `PROCESSING_REPORT.md` / `PROCESSING_REPORT.pdf` — full explanation of how the raw M5 files were audited and transformed into `sales_long_full.parquet`.
- `_audit/` — machine-readable evidence backing the processing and validation claims (quality audit, build stats, validation results, sample coverage/validation).
