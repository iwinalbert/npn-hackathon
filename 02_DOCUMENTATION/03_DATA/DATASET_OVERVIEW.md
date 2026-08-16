# 01_DATA — pointer, not a copy

**The datasets were deliberately NOT moved here.** They live at:

| Dataset | Canonical location | Status |
|---|---|---|
| Raw M5 competition CSVs (5 files) | `data/raw/` | **IMMUTABLE** — never opened in write mode |
| Processed long panel (59.2M rows) | `data/processed/sales_long_full.parquet` | build output, not regenerated |
| Inspection sample | `data/processed/sales_sample_50000.csv` | build output |
| Build + quality audits | `data/processed/_audit/` | build output |

## Why they were not moved

`pipeline/config.py` derives every path from

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
```

Moving `data/` would require editing `config.py`, which would change the file
every one of the 58 pipeline scripts imports, and would break reproduction of all
86 experiments. The task brief is explicit that reproducibility outranks folder
aesthetics, so the data stays put and this file points at it.

## Raw data integrity

MD5s recorded in `data/raw/README.md` and re-verified during the Stage 7 audit
and again during this reorganisation:

| File | MD5 |
|---|---|
| calendar.csv | 3ffeab2991b0c8e861d008b39ea4c95c |
| sales_train_evaluation.csv | b806dfc9f30a745102b708c09951f6aa |
| sales_train_validation.csv | 26a366a25beb57b0a8f4c7b148758f94 |
| sample_submission.csv | c281a69d7c011274899d92020a66e25b |
| sell_prices.csv | 08c591caa99e55daf3e0ccac913f7c85 |

A full SHA-256 manifest of all 520 protected files is in
`02_DOCUMENTATION/09_VALIDATION/_integrity/`.

## For the backend phase

Do **not** read the raw CSVs from a web request — the panel is 1.78 GB and takes
~15 s to load. Serve the **forecast** (`02_DOCUMENTATION/11_SUBMISSION/`), which is 30,490 rows and
loads instantly. Load reference data once at process start if you need item or
store metadata.
