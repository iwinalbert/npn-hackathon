# EDA Methodology

This document explains **how** the analysis behind `EDA_REPORT.md` / `EDA_REPORT.pdf` was performed, so results can be reproduced or extended. For the findings themselves, see `EDA_REPORT.md`. For raw machine-readable evidence, see `EDA/statistics/*.json` and `EDA/tables/*.csv`.

## Scope and ground rules

- **Read-only against `processed_dataset/sales_long_full.parquet`.** No script in this stage opens that file (or anything in `raw_dataset/`) in write mode.
- **No zeros removed, replaced, or reinterpreted.** Zero-sales rows are measured, not altered.
- **No outliers removed.** High sales values are treated as genuine observations throughout.
- **No promotion or stockout labels invented.** Where price changes or zero patterns look suggestive, this is stated as a hypothesis, never as fact.
- **Every number in the report traces to a script output.** Each script writes its computed statistics to `EDA/statistics/*.json`, its tables to `EDA/tables/*.csv`, and its charts to `EDA/charts/*.png` before the narrative report is written from those files.

## Tooling

- Python 3.13, pandas 3.0.5, numpy, pyarrow 25.0.1 (Parquet I/O), matplotlib 3.11.1 (charts), reportlab 5.0.0 (PDF).
- Chart colors follow the project's dataviz-skill reference palette (fixed categorical hue order, single-hue sequential ramp, no dual-axis charts, no pie/donut charts).

## Script-by-script approach

| Script | Phases | What it computes |
|---|---|---|
| `eda_01_sanity_and_distribution.py` | 1, 2 | Structural counts, missing values, sales percentiles, per-series and per-group (category/department/store/state) sales summaries, sales concentration curve. |
| `eda_02_zero_intermittent.py` | 3 | Zero% by group and per series; data-driven activity-class thresholds (p25/p75/p95 of the per-series zero% distribution); vectorized run-length encoding (RLE) of consecutive zero-sales streaks; leading-zero-block length vs. leading-no-price length comparison. |
| `eda_03_temporal.py` | 4 | Day-of-week, monthly, yearly, and week-of-year aggregates; daily total sales trend with rolling averages; a composition-effect check comparing yearly sales growth against cumulative active-series count. |
| `eda_04_events_snap.py` | 5, 6 | Event-day vs. non-event-day sales by group; per-named-event lift against a local same-weekday non-event baseline; state-matched SNAP indicator construction and SNAP-day vs. non-SNAP-day sales by group. |
| `eda_05_price.py` | 7 | Price distributions; price-change detection and frequency per series; within-series relative-price bins vs. mean sales; vectorized before/after rolling-window sales comparison around detected price-change events. |
| `eda_06_heterogeneity_correlation.py` | 8, 9 | Activity-class x volume-tier segmentation; per-series seasonality index (CV of monthly means); a documented stratified sample for per-series event/price sensitivity; full-data lag (1/7/14/28) and rolling-mean correlations with same-day sales; a vectorized "days since last sale" construction and its relationship to today's sale probability. |
| `eda_07_feature_table.py` | 10, 11 | Candidate feature table, each row grounded in a specific phase's evidence, with an explicit leakage-risk assessment per feature. |

## Key methodological decisions

**Sort order for time-dependent computations.** `sales_long_full.parquet`'s physical row order is date-major (an artifact of the original wide-to-long melt: all series for `d_1`, then all series for `d_2`, etc.), not id-major. Every computation that depends on chronological order within a series (zero-run streaks, lag features, rolling windows, price-change before/after windows) explicitly sorts by `(id, date)` first rather than assuming order.

**Zero-run detection.** Consecutive zero-sales streaks are detected via a fully vectorized run-length encoding: a new "run" starts wherever either the series id changes or the zero/non-zero status changes, compared position-by-position in the sorted array. This avoids a 30,490-way Python loop while remaining exact.

**Activity-class thresholds are data-driven, not arbitrary.** The four-way split of series into High-activity / Regular-intermittent / Sparse / Extremely-sparse uses the 25th, 75th, and 95th percentiles of *this dataset's own* per-series zero% distribution (computed once in Phase 3, reused identically in Phase 8) - not fixed round-number cutoffs chosen in advance.

**Event lift uses a local, weekday-matched baseline.** A named event's "lift" is not compared to the dataset's overall average (which would confound day-of-week and season), but to the mean sales of *other dates with the same weekday*, within a +/-21-day window, that are not themselves event days. This isolates the event's association from ordinary weekly and seasonal patterns as far as the available data allows.

**Relative price, not raw price, for within-item price sensitivity.** Raw price is dominated by cross-item scale differences (a $30 hobby item vs. a $1 food item). To look at price sensitivity in a way that isn't swamped by that, each row's price is expressed as a ratio to that series' own mean observed price before bucketing.

**Composition-effect check on the yearly trend.** Because the panel is fully rectangular in time (every series has a row for every day from 2011-01-29, even before it was actually stocked), a naive year-over-year sales trend partly reflects more items being listed over time, not organic growth. Phase 4 explicitly compares total yearly sales against the cumulative count of series with an observed price by that year's end (using the same "first priced date" signal as Phase 3) to make this composition effect visible rather than silently baked into a trend narrative.

**Sampling was used exactly once, and is documented where it appears.** Every phase runs on the complete 59.18M-row table except one sub-analysis in Phase 8 (per-series event sensitivity and price-sales correlation), which uses a stratified random sample (seed=99, n=501, proportional across category x volume-tier strata) because computing an individual regression/correlation for all 30,490 series individually was unnecessary for an exploratory distribution-of-effects view. This is the only sampling in the entire EDA; it is called out explicitly in both `EDA/statistics/audit_metadata.json` and the report itself.

## Reproducing this analysis

1. Confirm `processed_dataset/sales_long_full.parquet` matches the checksum recorded in `EDA/statistics/audit_metadata.json` / the Final Validation section of `EDA_REPORT.md`.
2. Run the seven `eda_*.py` scripts listed above in order (each is independent except that Phase 8's segmentation step reads two CSVs written by Phase 2 and Phase 3, and Phase 4 reads one CSV written by Phase 3).
3. All outputs are deterministic given the fixed random seeds documented above and in `audit_metadata.json`.
