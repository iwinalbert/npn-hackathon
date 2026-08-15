"""
Central configuration for the M5 forecasting pipeline.

Everything that another module might want to tweak (paths, the forecast horizon,
which day we treat as "today" during backtesting) lives here, so no module has to
hard-code a path or a magic number.

READ-ONLY GUARANTEE
-------------------
This pipeline never opens anything under raw_dataset/ or processed_dataset/ in
write mode. Those folders are inputs only. Everything this pipeline produces is
written to artifacts/, reports/, models/, experiments/ or predictions/.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Source data (IMMUTABLE) ---------------------------------------------
# data/raw/ holds the five original competition CSVs and nothing else. No code
# in this repository opens anything under it in write mode.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SALES_EVAL_CSV = RAW_DIR / "sales_train_evaluation.csv"
SALES_VALID_CSV = RAW_DIR / "sales_train_validation.csv"
CALENDAR_CSV = RAW_DIR / "calendar.csv"
SELL_PRICES_CSV = RAW_DIR / "sell_prices.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

PROCESSED_PARQUET = PROCESSED_DIR / "sales_long_full.parquet"

# --- Pipeline outputs -----------------------------------------------------
# experiments/registry/  one JSON record per experiment (the ledger)
# experiments/artifacts/ result tables and diagnostics those runs produced
# models/experiments/    models produced by experimental runs
# models/champion/       the selected model, curated by hand — never written to
# predictions/validation/     backtest predictions, one file per experiment
# predictions/final_forecast/ the d_1942-d_1969 deliverable, curated by hand
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR = EXPERIMENTS_ROOT / "registry"
ARTIFACTS_DIR = EXPERIMENTS_ROOT / "artifacts"

MODELS_ROOT = PROJECT_ROOT / "models"
MODELS_DIR = MODELS_ROOT / "experiments"
CHAMPION_DIR = MODELS_ROOT / "champion"

PREDICTIONS_ROOT = PROJECT_ROOT / "predictions"
PREDICTIONS_DIR = PREDICTIONS_ROOT / "validation"
FINAL_FORECAST_DIR = PREDICTIONS_ROOT / "final_forecast"

# Reports are filed by project stage (reports/01_foundation, 02_modelling, ...).
# A regenerated report is written to the reports/ root and should then be filed
# into the stage folder it belongs to.
REPORTS_DIR = PROJECT_ROOT / "reports"

DOCS_DIR = PROJECT_ROOT / "docs"

for _d in (ARTIFACTS_DIR, REPORTS_DIR, MODELS_DIR, CHAMPION_DIR, EXPERIMENTS_DIR,
           PREDICTIONS_DIR, FINAL_FORECAST_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Dataset shape constants
#
# These are NOT assumptions — every one of them was independently verified
# against raw_dataset/ (see FINAL_APPROACH/SUPPORTING_EVIDENCE.md). The loader
# asserts against them at run time, so if a source file ever changes, the
# pipeline fails loudly instead of silently producing wrong numbers.
# --------------------------------------------------------------------------

N_SERIES = 30_490          # store-item combinations = 3,049 items x 10 stores
N_ITEMS = 3_049
N_STORES = 10
N_STATES = 3
N_DEPTS = 7
N_CATS = 3

N_HISTORY_DAYS = 1_941     # d_1 .. d_1941  (2011-01-29 .. 2016-05-22)
N_CALENDAR_DAYS = 1_969    # d_1 .. d_1969  (calendar runs 28 days past the sales)
N_PRICE_WEEKS = 282        # distinct wm_yr_wk values

# Known-good totals, re-verified directly from the raw CSVs.
# Used by validation_checks.py as a data-integrity fingerprint.
EXPECTED_TOTAL_UNITS = 66_927_173
EXPECTED_ZERO_CELLS = 40_241_819
EXPECTED_MAX_SALES = 763


# --------------------------------------------------------------------------
# Forecasting task definition
# --------------------------------------------------------------------------

HORIZON = 28               # we forecast 28 days ahead, per sample_submission.csv (F1..F28)

# Day indices below are ZERO-BASED positions into the day axis:
#   index 0   == "d_1"    == 2011-01-29
#   index 1940 == "d_1941" == 2016-05-22   (last day with known sales)
#   index 1968 == "d_1969" == 2016-06-19   (last day in calendar.csv)

LAST_KNOWN_DAY_IDX = N_HISTORY_DAYS - 1            # 1940 -> d_1941 -> 2016-05-22

# The real, genuinely-unknown forecast window: d_1942 .. d_1969.
# No file anywhere contains actual sales for these days.
FINAL_FORECAST_ORIGIN_IDX = LAST_KNOWN_DAY_IDX     # 1940

# The primary BACKTEST window.
#
# We cut training off at d_1913 (2016-04-24) and validate against d_1914..d_1941
# (2016-04-25 .. 2016-05-22). Those 28 days are real, already-observed sales that
# live in sales_train_evaluation.csv but NOT in sales_train_validation.csv, which
# makes them a natural held-out block: it reproduces the exact shape of the real
# task (28 days ahead from a fixed origin) using days whose true answers we have.
VALIDATION_ORIGIN_IDX = 1912                       # d_1913 -> 2016-04-24

# Additional origins usable for rolling-origin backtesting later (each is a
# further 28 days back). Kept here so the choice is explicit and reviewable
# rather than buried in a script.
ALT_VALIDATION_ORIGIN_IDXS = [1912 - 28, 1912 - 56, 1912 - 84]


# --------------------------------------------------------------------------
# Feature-engineering parameters
# --------------------------------------------------------------------------

LAGS = [1, 7, 14, 28]              # origin-relative lags (see features.py for the exact definition)
ROLLING_WINDOWS = [7, 28]          # windows for rolling mean / std, ending at the origin
PRICE_AVG_WINDOW_WEEKS = 8         # trailing window for "recent average price"

RANDOM_SEED = 42
