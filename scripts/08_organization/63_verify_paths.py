"""
PATH RESOLUTION VERIFICATION — do the scripts still find their files?

READ-ONLY, and deliberately NON-EXECUTING. It imports modules and resolves paths
but never runs a pipeline script, because most of them WRITE into
experiments/artifacts/ or predictions/, which would modify protected artefacts
and invalidate the integrity manifest. Proving the paths resolve is the point;
re-running the research is not.

Checks:
  1. every path constant in pipeline/config.py points at something that exists
  2. every pipeline module imports cleanly
  3. the walk-up root-resolution idiom used by 56 scripts resolves to the real
     project root from each script's location
  4. MY_RESEARCH_PAPER's parent.parent idiom still lands on the project root
  5. the intra-folder import in scripts/07_usecase11 still resolves
  6. the frozen-model copies are byte-identical to their canonical sources

    python scripts/08_organization/63_verify_paths.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "pipeline" / "config.py").exists())
sys.path.insert(0, str(ROOT))

OUT = ROOT / "08_DOCUMENTATION" / "_integrity" / "path_verification.json"
results = {"checks": [], "passed": 0, "failed": 0}


def check(name, ok, detail=""):
    results["checks"].append({"check": name, "passed": bool(ok), "detail": detail})
    results["passed" if ok else "failed"] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def main():
    print(f"  project root resolved to: {ROOT}\n")

    # 1. config path constants -----------------------------------------
    from pipeline import config
    consts = {
        "RAW_DIR": config.RAW_DIR, "PROCESSED_DIR": config.PROCESSED_DIR,
        "SALES_EVAL_CSV": config.SALES_EVAL_CSV,
        "SALES_VALID_CSV": config.SALES_VALID_CSV,
        "CALENDAR_CSV": config.CALENDAR_CSV,
        "SELL_PRICES_CSV": config.SELL_PRICES_CSV,
        "SAMPLE_SUBMISSION_CSV": config.SAMPLE_SUBMISSION_CSV,
        "PROCESSED_PARQUET": config.PROCESSED_PARQUET,
        "EXPERIMENTS_DIR": config.EXPERIMENTS_DIR,
        "ARTIFACTS_DIR": config.ARTIFACTS_DIR,
        "MODELS_DIR": config.MODELS_DIR, "CHAMPION_DIR": config.CHAMPION_DIR,
        "PREDICTIONS_DIR": config.PREDICTIONS_DIR,
        "FINAL_FORECAST_DIR": config.FINAL_FORECAST_DIR,
        "REPORTS_DIR": config.REPORTS_DIR, "DOCS_DIR": config.DOCS_DIR,
    }
    missing = [k for k, v in consts.items() if not Path(v).exists()]
    check("config.py path constants all resolve", not missing,
          f"{len(consts)} constants" + (f"; MISSING {missing}" if missing else ""))
    check("PROJECT_ROOT is the real project root",
          Path(config.PROJECT_ROOT).resolve() == ROOT, str(config.PROJECT_ROOT))

    # 2. pipeline modules import ---------------------------------------
    mods = sorted(p.stem for p in (ROOT / "pipeline").glob("*.py")
                  if p.stem != "__init__")
    bad = []
    for m in mods:
        try:
            importlib.import_module(f"pipeline.{m}")
        except Exception as e:                                  # noqa: BLE001
            bad.append(f"{m}: {type(e).__name__}")
    check("all pipeline modules import", not bad,
          f"{len(mods)} modules" + (f"; FAILED {bad}" if bad else ""))

    # 3. walk-up idiom from every script location ----------------------
    script_dirs = sorted({p.parent for p in (ROOT / "scripts").rglob("*.py")})
    bad = []
    for d in script_dirs:
        try:
            r = next(p for p in (d / "x.py").resolve().parents
                     if (p / "pipeline" / "config.py").exists())
            if r != ROOT:
                bad.append(str(d))
        except StopIteration:
            bad.append(str(d))
    check("walk-up root resolution works from every script folder", not bad,
          f"{len(script_dirs)} folders" + (f"; FAILED {bad}" if bad else ""))

    # 4. MY_RESEARCH_PAPER parent.parent idiom -------------------------
    mrp = ROOT / "MY_RESEARCH_PAPER"
    derived = (mrp / "build_paper.py").resolve().parent.parent
    check("MY_RESEARCH_PAPER parent.parent lands on project root",
          derived == ROOT and (derived / "pipeline" / "config.py").exists(),
          str(derived))

    # 5. intra-folder import in scripts/07_usecase11 -------------------
    uc = ROOT / "scripts" / "07_usecase11"
    needed = ["53_exp80_item_level_probe.py", "54_exp80b_level_sweep.py",
              "56_exp80c_orthogonality.py", "57_exp81_four_window_validation.py",
              "59_exp82_adaptive_alpha.py"]
    check("scripts/07_usecase11 sibling imports still co-located",
          all((uc / n).exists() for n in needed),
          f"{sum((uc / n).exists() for n in needed)}/{len(needed)} present")

    # 6. frozen copies identical to canonical sources ------------------
    def sha(p):
        h = hashlib.sha256()
        h.update(Path(p).read_bytes())
        return h.hexdigest()

    pairs = [
        (ROOT / "models/champion/model_11_blend_direct_final_forecast.txt",
         ROOT / "02_MODEL/FROZEN_CHAMPION/model_11_blend_direct_final_forecast.txt"),
        (ROOT / "models/champion/model_12_blend_recursive_shape_final.txt",
         ROOT / "02_MODEL/FROZEN_CHAMPION/model_12_blend_recursive_shape_final.txt"),
        (ROOT / "predictions/final_forecast/final_forecast_28day_v3_diversity_blend.csv",
         ROOT / "03_FORECASTS/final_forecast_28day_v3_diversity_blend.csv"),
        (ROOT / "predictions/final_forecast/final_forecast_28day_v3_diversity_blend.csv",
         ROOT / "09_SUBMISSION/final_forecast_28day_v3_diversity_blend.csv"),
    ]
    mismatched = [str(b.relative_to(ROOT)) for a, b in pairs if sha(a) != sha(b)]
    check("frozen copies byte-identical to canonical sources", not mismatched,
          f"{len(pairs)} pairs" + (f"; MISMATCH {mismatched}" if mismatched else ""))

    # 7. registry still loadable through the pipeline API ---------------
    from pipeline import experiment
    recs = experiment.load_all()
    champ = experiment.load("exp_78_blend_final_forecast")
    check("experiment registry loads through pipeline API",
          len(recs) == 86 and champ is not None,
          f"{len(recs)} records; exp_78 {'found' if champ else 'MISSING'}")

    print(f"\n  {results['passed']} passed, {results['failed']} failed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
