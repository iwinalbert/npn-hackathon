# Supporting Evidence — Numeric Audit Trail

This file is the traceability backbone for `FINAL_PROJECT_APPROACH.md`. For every important numerical claim used in that document, this file states: where it came from, whether it was independently re-verified for this review, the exact verification method, and — for the four previously-disputed numbers — the full resolution.

**Scope note:** all verification below was performed read-only against `raw_dataset/sales_train_evaluation.csv`, `raw_dataset/sales_train_validation.csv`, `raw_dataset/calendar.csv`, and `processed_dataset/sales_long_full.parquet`, using Python 3 / pandas 3.0.5 / pyarrow 25.0.1 (matching the environment already documented in `EDA/statistics/audit_metadata.json`). No file was opened in write mode. No file was modified, moved, or deleted.

---

## 1. Independent re-verification — raw `sales_train_evaluation.csv`

**Command (structure/counts):**
```python
import pandas as pd, numpy as np
sales = pd.read_csv("raw_dataset/sales_train_evaluation.csv")
d_cols = [c for c in sales.columns if c.startswith("d_")]
```

**Output:**
```
Rows (series): 30490
Day columns: 1941 d_1 to d_1941
Unique item_id: 3049
Unique store_id: 10
Unique id: 30490
Unique dept_id: 7
Unique cat_id: 3
Unique state_id: 3

Total cells (rows x days): 59181090
Max sales value: 763
Min sales value: 0
Sum of all sales: 66927173
Zero count: 40241819  pct: 67.99776584040612
Max value row id: FOODS_3_090_CA_3_evaluation  day: d_960

Category sums:
FOODS        45922427
HOBBIES        6240656
HOUSEHOLD    14764090

FOODS share pct: 68.61551884165196
```

**Cross-check against `calendar.csv` for the record-sale date:**
```
d = d_960 -> date = 2013-09-14, weekday = Saturday, event_name_1 = NaN, snap_CA = 0
```

**Agreement with existing reports:** exact match to `PROCESSING_REPORT.md` §11/§14 (59,181,090 rows; sum 66,927,173; zero count 40,241,819; max 763) and to `EDA_REPORT.md` §3 (identical figures, stated as unchanged since processing).

---

## 2. Independent re-verification — raw `sales_train_validation.csv` (cross-check file)

**Output:**
```
Validation file max: 763
Rows: 30490
Days: 1913
Cells: 58327370

Validation-file category sums:
FOODS        45089939
HOBBIES        6124800
HOUSEHOLD    14480670

FOODS share (validation file, units): 68.63484022148336
```

Confirms the maximum (763) and FOODS share (~68.6%) are stable whether computed on the shorter validation-file history or the full evaluation-file history — not an artifact of the extra 28 evaluation-only days.

---

## 3. Independent re-verification — processed `sales_long_full.parquet`

**Output:**
```
Processed parquet shape check:
Rows: 59181090
Total units: 66927173
cat_id
FOODS        45922427
HOBBIES        6240656
HOUSEHOLD    14764090
FOODS unit share (parquet): 68.61551884165196
```

Exact match to the raw-file recomputation in Section 1 — confirms the melt + calendar join + price join pipeline documented in `PROCESSING_REPORT.md` introduced no drift in row count, sales sum, zero count, max, or category totals.

---

## 4. Resolution — 634 vs. 763 (maximum single-day sales)

**Question:** is 634 a competing, unresolved figure for the dataset's true maximum?

**Finding:** No. 634 is a real number that exists in the data, but it is a *department-scoped* and (coincidentally) *store-scoped* maximum, not the dataset-wide maximum.

**Command / output:**
```python
dept_max = sales.groupby('dept_id')[d_cols].max().max(axis=1)
```
```
dept_id
FOODS_1        166
FOODS_2        634   <-- source of "634"
FOODS_3        763   <-- true dataset-wide max lives here
HOBBIES_1      294
HOBBIES_2      135
HOUSEHOLD_1    626
HOUSEHOLD_2    601
```
```python
store_max = sales.groupby('store_id')[d_cols].max().max(axis=1)
```
```
store_id
CA_1    648
CA_2    227
CA_3    763   <-- true dataset-wide max lives here
CA_4    300
TX_1    634   <-- also coincides with "634"
TX_2    626
TX_3    385
WI_1    224
WI_2    300
WI_3    374
```

**Top 5 raw values overall** (confirms 763 is not an isolated fluke and there is a genuine cluster of large spikes near the top):
```
[648, 693, 709, 709, 763]
```

**Verdict:** 763 is correct and belongs to series `FOODS_3_090_CA_3`, department `FOODS_3`, store `CA_3`, on `d_960` = 2013-09-14. 634 is simply the maximum *within* department FOODS_2 (or, separately, within store TX_1) — a real but narrower-scoped statistic that was at some point reported without the scope attached.

---

## 5. Resolution — ~30M vs. 59,181,090 (processed row count)

**Finding:** 59,181,090 = 30,490 × 1,941 exactly (verified by direct multiplication and by `len()` on both the melted raw structure and the actual processed Parquet file). A full-text search of every `.md` and `.json` file in this project for "30M", "~30 million", "30,000,000" found **zero occurrences** — the "~30M" figure does not exist as a written claim anywhere in this project's own documentation. It is treated in `FINAL_PROJECT_APPROACH.md` as an unrecorded verbal/back-of-envelope figure, most plausibly a units confusion between the *series count* (30,490) and a *row count*.

**Verdict:** 59,181,090 is exact and independently confirmed three separate ways (raw structural count, raw recomputation, processed Parquet read). Use it, not an approximation.

---

## 6. Resolution — 69.56% vs. 68.6% (FOODS share of total sales)

**Finding:** every definition tested converges on ~68.6%, and none reproduces 69.56%.

| Definition tested | Result |
|---|---|
| Unit share, `sales_train_evaluation.csv` (full history) | 68.6155% |
| Unit share, `sales_train_validation.csv` (shorter history) | 68.6348% |
| Unit share, `sales_long_full.parquet` | 68.6155% |
| Revenue-weighted share (units × sell_price, `sales_long_full.parquet`) | 58.0131% |
| Share of nonzero rows (row-count-weighted, not unit-weighted) | 56.2969% |

**Command (revenue-weighted check):**
```python
df = pd.read_parquet("processed_dataset/sales_long_full.parquet", columns=['cat_id','sales','sell_price'])
df['revenue'] = df['sales'] * df['sell_price']
rev_by_cat = df.groupby('cat_id', observed=True)['revenue'].sum()
```
```
FOODS        111140024.0
HOBBIES        23321644.0
HOUSEHOLD      57115876.0
FOODS revenue share: 58.013077%
```

**Verdict:** 68.6% (68.62% to two decimal places) is correct and robust across three independent recomputations. 69.56% does not match any tested definition of "FOODS share" on this dataset and should be treated as an unverified/incorrect figure — drop it from all future materials.

---

## 7. Resolution — 42,840 vs. 30,490 (series count)

**Finding:** both numbers are individually correct; they are not in conflict, they answer different questions.

- **30,490** — independently re-verified: `sales['id'].nunique() == 30490`; `item_id.nunique() == 3049`; `store_id.nunique() == 10`; every item confirmed present in exactly 10 stores (a perfectly balanced 3,049 × 10 grid, no missing combinations). This is the row-count requirement of `sample_submission.csv` (60,980 = 30,490 × 2 ID-suffix blocks) and the actual forecasting granularity for this hackathon.
- **42,840** — the sum of series counts across all 12 official M5 WRMSSE hierarchy levels: 1 (grand total) + 3 (state) + 10 (store) + 3 (category) + 7 (department) + 9 (state×category) + 21 (state×department) + 30 (store×category) + 70 (store×department) + 3,049 (item) + 9,147 (state×item) + 30,490 (store×item):
```
1 + 3 + 10 + 3 + 7 + 9 + 21 + 30 + 70 + 3049 + 9147 + 30490 = 42840
```
This arithmetic identity is confirmed and matches the well-documented, public structure of the M5 Forecasting — Accuracy competition's official evaluation hierarchy. It is background/domain knowledge about the public competition this dataset is drawn from, not a number independently computed from this project's own files, and is analogous to how `ML_FORECASTING_APPROACH.md` already flags WRMSSE itself as "general background knowledge... not confirmed against our own hackathon's instructions."

**Verdict:** 30,490 is the number to build, train, and forecast for. 42,840 is only relevant if the team implements the full 12-level hierarchical reconciliation / official WRMSSE metric, which is not required unless the hackathon's own scoring rubric asks for it (unconfirmed — see "Team Decisions Required" in `ML_FORECASTING_APPROACH.md`, still open).

---

## 8. Cross-report consistency checks (not independently recomputed, but internally verified)

| Claim | Source(s) | Consistency check |
|---|---|---|
| Zero-sales rate 68.00% / 40,241,819 rows | `PROCESSING_REPORT.md`, `EDA_REPORT.md`, this review | Independently recomputed in Section 1 above — exact match |
| Missing `sell_price` 20.78% / 12,299,413 rows | `PROCESSING_REPORT.md` §10 | Stated once, at the join stage; not recomputed for this review, but internally consistent with the 64.1% of *series* having partial price coverage from `DATASET_SUMMARY.md` §11 (series-level vs. row-level are different denominators, not competing numbers) |
| Leading-zero / leading-no-price alignment (99.48% within 7 days, median gap 3 days) | `EDA_REPORT.md` §6, Phase 3 | Not independently recomputed (would require reproducing the full run-length-encoding + first-priced-date join); accepted on the strength of the EDA's documented methodology (`EDA_METHODOLOGY.md`, explicit `(id, date)` sort, vectorized RLE) and its consistency with the independently-confirmed example in `DATASET_SUMMARY.md` §14 (`FOODS_3_595` / `CA_1`: 1,841 leading zero days, price starts at week 11603) |
| rolling_mean_7 r=0.820, days_since_last_sale staircase (65.2%→0.6%) | `EDA_REPORT.md` §12, Phase 9 | Not independently recomputed for this review (large correlation computation over 59M rows); accepted on the strength of documented full-data methodology (no sampling in Phase 9, per `audit_metadata.json`) |
| SNAP effect (+12.7% overall, +17.3% FOODS, +32.3% FOODS_2) | `EDA_REPORT.md` §9, Phase 6 | Not independently recomputed; internally consistent with the independent "~10% CA SNAP lift" spot-check already noted in `DATASET_SUMMARY.md` §8 (an earlier, cruder check from the raw-investigation stage, same direction and rough magnitude) |
| Named event effects (Christmas −99.95%, LaborDay +27.5%) | `EDA_REPORT.md` §8, Phase 5 | Not independently recomputed; methodology (local weekday-matched ±21-day baseline) is documented and sound |

Figures in this table were not independently recomputed for this review (doing so would require reproducing multi-hundred-line analysis scripts against the full 59M-row table) — they are accepted because their source methodology is transparently documented (`EDA_METHODOLOGY.md`) and because every figure that *was* re-derivable from first principles (Sections 1–7 above) matched exactly. No contradiction was found anywhere between what the existing reports claim and what this review could independently verify.

---

## 9. What was explicitly searched for and NOT found

A full-text search (`grep`) across every file in the project for the literal strings `634`, `42,840` / `42840`, `69.56`, `~30M` / `30 million` / `30,000,000` found no report-level claim matching any of these numbers as an assertion about the dataset (the only "634"/"634" pattern hits were incidental digit sequences inside unrelated raw data rows/prices, not stated claims). This supports the conclusion in `FINAL_PROJECT_APPROACH.md` Step 1 that these four figures most likely originated in team discussion or an early draft that is not part of the committed project files, rather than from a genuine second analysis script that computed something different.

---

*This file supports `FINAL_PROJECT_APPROACH.md`. No file in `raw_dataset/`, `processed_dataset/`, `EDA/`, `Project_Approach/`, or `analysis_output/` was modified while producing it.*
