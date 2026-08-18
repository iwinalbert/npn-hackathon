# Repository Organization Report

*Executed 2026-08-14. 357 files relocated, 0 files lost, 0 protected bytes changed.*

> **Outcome:** the repository was reorganized from 14 loosely-named top-level folders into a conventional structure, with every protected artefact verified byte-identical afterwards and the pipeline confirmed still functional by a live run.

---

## 1. Original structure

Fourteen top-level directories with inconsistent naming conventions (`raw_dataset`, `EDA`, `analysis_output`, `Project_Approach`, `FINAL_APPROACH`, `ProblemStatement_Walkthrough`), documentation spread across five of them, 31 scripts in one flat folder, 71 experiment records and 36 artifacts intermixed, 50 report files unfiled, and the final forecast sitting alongside 26 validation prediction files.

```
raw_dataset/        processed_dataset/   EDA/            analysis_output/
Project_Approach/   FINAL_APPROACH/      ProblemStatement_Walkthrough/
pipeline/  scripts/  experiments/  models/  predictions/  artifacts/  reports/
end_to_end_approach.md   requirements.txt
```

## 2. Final structure

```
README.md  PROJECT_INDEX.md  requirements.txt  .gitignore
data/raw/            the 5 original CSVs — IMMUTABLE
data/processed/      sales_long_full.parquet + build/audit reports
pipeline/            reusable source package (13 modules)
scripts/             6 stage folders, chronological numbering preserved
experiments/registry/    71 JSON records
experiments/artifacts/   36 result tables and diagnostics
experiments/EXPERIMENT_LEDGER.md
models/champion/     the selected model + its forecast retrain
models/experiments/  10 other experimental models
predictions/final_forecast/  the deliverable
predictions/validation/      26 backtest prediction files
reports/             5 stage folders + charts/
docs/                5 numbered folders, chronological
```

## 3. Major organizational decisions

| Decision | Reasoning |
|---|---|
| `data/raw/` holds **only** the 5 source CSVs | The dataset guides that lived inside `raw_dataset/Dataset_Explanation/` are documentation, not data. Moving them to `docs/02_dataset/` makes the raw folder purely immutable source. |
| Kept the package named `pipeline/` rather than renaming to `src/` | It is descriptive and already correct; renaming would have touched 31 import statements for no functional gain. |
| Experiment records kept **flat** in `registry/` | The 71 JSONs are a ledger, not 71 projects. Seventy-one folders each holding one file would be worse navigation. `EXPERIMENT_LEDGER.md` provides the index instead. |
| Artifacts kept flat in `experiments/artifacts/` | Scripts write here via `config.ARTIFACTS_DIR`; subfoldering would break write paths on re-run. An in-place README maps every file to the run that produced it. |
| Final forecast separated from validation predictions | The deliverable should never be confused with 26 backtest files. |
| Champion separated from other models | `models/champion/` is curated by hand; new training runs write to `models/experiments/`. |
| Reports filed by **stage**, md+pdf pairs together | Grouping by format would separate each report from its source. |
| Script bootstrap made depth-independent | Scripts now locate the project root by walking up to the folder containing `pipeline/config.py`, so they keep working wherever they are filed. |

## 4. Files moved

**357 files** relocated. By destination:

| Destination | Files |
|---|---|
| `experiments/registry/` | 71 |
| `docs/04_eda/tables/` | 38 |
| `experiments/artifacts/` | 36 |
| `docs/04_eda/charts/` | 26 |
| `predictions/validation/` | 26 |
| `docs/03_exploratory_analysis/` | 24 |
| `reports/04_optimization/` | 22 |
| `reports/02_modelling/` | 16 |
| `models/experiments/` | 10 |
| `docs/04_eda/statistics/` | 9 |
| `scripts/04_optimization/` | 9 |
| `docs/05_approach/` | 7 |
| `scripts/02_modelling/` | 7 |
| `reports/05_diagnostics_and_research/` | 6 |
| `scripts/03_benchmark_investigation/` | 6 |
| `data/raw/` | 5 |
| `docs/02_dataset/` | 5 |
| `data/processed/_audit/` | 5 |

The complete move log — every source path and destination path — is preserved verbatim in `experiments/artifacts/repository_move_log.json`.

## 5. Files renamed

| From | To | Why |
|---|---|---|
| `end_to_end_approach.md` (project root) | `docs/01_problem_statement/TEAM_end_to_end_approach.md` | It is the other team's document; the prefix prevents it being mistaken for ours, and it no longer clutters the root. |

No other file was renamed. Experiment ids, model filenames and prediction filenames were left exactly as they were, so every reference inside the 71 JSON records and 25 reports still resolves by name.

## 6. Files removed

| Removed | Count | Why it is safe |
|---|---|---|
| `pipeline/__pycache__/` | 1 directory | Compiled bytecode, regenerated automatically on next import |
| Empty source directories | 13 | Left behind after their contents moved; contained nothing |

**Nothing else was deleted.** In particular, four groups of byte-identical model and prediction files were found and **deliberately kept**: they are independent re-runs of the same configuration that reproduce identical output, which is the project's evidence that the pipeline is deterministic. Removing them as "duplicates" would have destroyed that evidence.

## 7. Protected file integrity — before vs after

| File (final location) | MD5 | Match |
|---|---|---|
| `data/raw/calendar.csv` | `3ffeab2991b0c8e861d008b39ea4c95c` | MATCH |
| `docs/02_dataset/Cognizant_M5_Problem_Approach_Algorithms_Hierarchical_Plan.pdf` | `886d7de7ce7e725a48f014ecaaa3dbe0` | MATCH |
| `docs/02_dataset/Cognizant_Walmart_M5_Dataset_Deep_Explanation.pdf` | `9b29d38c7c593793579dbdf8e8099a7d` | MATCH |
| `docs/02_dataset/DATASET_EXPLAINED.md` | `f55aaab94fbe61d35a17d2a26404ba1b` | MATCH |
| `docs/02_dataset/DATASET_EXPLAINED.pdf` | `cebeea524f08cee7609f1e39a3248c3c` | MATCH |
| `docs/02_dataset/DATASET_SUMMARY.md` | `04d6961a5581bb3f04b3b4d670533a0b` | MATCH |
| `data/raw/sales_train_evaluation.csv` | `b806dfc9f30a745102b708c09951f6aa` | MATCH |
| `data/raw/sales_train_validation.csv` | `26a366a25beb57b0a8f4c7b148758f94` | MATCH |
| `data/raw/sample_submission.csv` | `c281a69d7c011274899d92020a66e25b` | MATCH |
| `data/raw/sell_prices.csv` | `08c591caa99e55daf3e0ccac913f7c85` | MATCH |
| `data/processed/sales_long_full.parquet` | `faa70ced1dddd2a84801a748b1149986` | MATCH |
| `predictions/final_forecast/final_forecast_28day.csv` | `d1067b9ff7a09fefadac71be27b7044c` | MATCH |
| `predictions/final_forecast/submission_m5_format.csv` | `1722effc629d8d74000c4c33c7ae36ca` | MATCH |
| `predictions/validation/model_04_tweedie_recency_listing_validation.csv` | `36553293c0d30e15b60f712a17dd4721` | MATCH |
| `models/champion/model_04_tweedie_recency_listing.txt` | `e1fb7b8733b0e030174fed3b2648f16a` | MATCH |
| `models/champion/model_07_final_forecast.txt` | `d722e2cc6ae11ae6f6d78ca7322a50e9` | MATCH |

**All 16 protected files are byte-identical to their pre-reorganization state: True.**

## 8. Integrity checks

| # | Check | Result |
|---|---|---|
| 1 | Raw dataset hashes unchanged | PASS (5/5) |
| 2 | Final forecast file unchanged | PASS |
| 3 | Champion validation predictions unchanged | PASS |
| 4 | Champion RMSE | 2.121043 — PASS |
| 5 | Champion MAE | 1.031927 — PASS |
| 6 | All 71 experiments present and parsing | 71/71 — PASS |
| 7 | All reports accessible and opening | 25 PDFs — PASS |
| 8 | No broken internal paths | PASS — 32 scripts compile; live data load, feature build and champion prediction all succeeded |
| 9 | No duplicate final artefacts created | PASS — `final_forecast/` holds exactly 2 files, `champion/` exactly 2 models |
| 10 | Repository navigable from root | PASS — README.md + PROJECT_INDEX.md |
| 11 | Final model unambiguously identifiable | PASS — `models/champion/` with an in-place README explaining the duplicate experiment ids |
| 12 | No experiment modified by the cleanup | PASS — spot-checked metrics across Experiments #4, #69, #70, #71 |

Check 8 was verified functionally, not just by inspection: after the move, `M5Data()` loaded the raw CSVs, `FeatureBuilderV2` built a validation frame, and the champion model file loaded and produced predictions.

## 9. Ambiguous items left for review

| Item | Where it is | Note |
|---|---|---|
| `docs/05_approach/_build_pdf.py` | with its document | A one-off renderer for `FINAL_PROJECT_APPROACH.md`, superseded by `pipeline/report_pdf.py`. Kept beside the document it builds because it resolves paths relative to its own location. Safe to delete if that document is never rebuilt. |
| `docs/03_exploratory_analysis/` | 24 files | First-pass exploration that predates the formal EDA. Superseded in substance but referenced by `docs/02_dataset/DATASET_SUMMARY.md`, so retained. |
| `.claude/settings.local.json` | project root | Tooling configuration, left untouched. |

Nothing was placed in a quarantine folder — every file had a clear home.

## 10. Entry points

| Purpose | Path |
|---|---|
| Understand the project | `README.md` |
| Find any artefact | `PROJECT_INDEX.md` |
| Navigate the 71 experiments | `experiments/EXPERIMENT_LEDGER.md` |
| Verify integrity and leakage | `scripts/01_foundation/01_foundation_check.py` |
| Regenerate the forecast | `scripts/02_modelling/08_final_forecast.py` |
| The deliverable | `predictions/final_forecast/final_forecast_28day.csv` |

---

*Repository organization only. No model was retrained, no parameter changed, no prediction value altered, and no experiment conclusion revised.*