# ORGANIZATION AUDIT

**Task:** organize the project before the backend/frontend phase, without
touching the frozen model, the datasets, the predictions or the experiment record.

**Result:** a new numbered delivery layer was added on top of the existing
research pipeline. **Nothing was moved, deleted or overwritten.** Verified by
SHA-256 over 520 protected files, before and after.

---

## 1. Headline verification

| Check | Result |
|---|---|
| Protected files fingerprinted before | **520** files, 1.78 GB |
| Protected files after | 521 (the +1 is `scripts/08_organization/62_experiment_classification.py`, written during this task) |
| Files **deleted** | **0** |
| Files **modified / overwritten** | **0** |
| Raw-data MD5s (5 files) | **all identical** to the recorded values |
| Frozen model copies vs canonical sources | **byte-identical** (SHA-256, 4 pairs) |
| Path-resolution checks | **8 / 8 passed** |
| Experiment registry loads through the pipeline API | 86 records, `exp_78` resolvable |

Evidence: `_integrity/manifest_before.json`, `_integrity/manifest_after.json`,
`_integrity/integrity_comparison.json`, `_integrity/path_verification.json`.

---

## 2. The central decision: two layers, not one relocation

**Nothing in the research pipeline could be physically moved.** This was
established *before* touching anything, by reading the code rather than assuming.

`pipeline/config.py`:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR   = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
PREDICTIONS_ROOT = PROJECT_ROOT / "predictions"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR   = PROJECT_ROOT / "docs"
```

Consequences:

- `pipeline/` must stay **exactly one level** below the project root.
- `data/`, `models/`, `predictions/`, `experiments/`, `reports/`, `docs/` must
  keep their exact names and positions.
- `MY_RESEARCH_PAPER/` has the same constraint via its own
  `ROOT = Path(__file__).resolve().parent.parent`, used by all five of its build
  scripts, which then `import pipeline`. Moving it into `05_REPORTS/` would make
  `ROOT` point at `05_REPORTS/`, which has no `pipeline/`, and every build script
  would fail on import.

Moving any of these would require editing `config.py` — which the brief
explicitly forbids ("do not alter scripts merely to make paths prettier") and
which would break reproduction of all 86 experiments.

**So the numbered folders are a delivery and navigation layer holding copies and
pointers, not a relocation.** This is the trade the brief asked for: the existing
research pipeline outranks folder aesthetics.

---

## 3. What was CREATED

| Path | Contents |
|---|---|
| `01_DATA/README.md` | pointer to `data/` + raw MD5 table + backend guidance |
| `02_MODEL/MODEL_FREEZE.md` | **the freeze document** — architecture, hyperparameters, metrics, leakage status, validation protocol, limitations, change control |
| `02_MODEL/README.md` | navigation + warning about the 3 superseded champions |
| `02_MODEL/FROZEN_CHAMPION/CHAMPION_MANIFEST.json` | SHA-256 + provenance of the frozen artefacts |
| `03_FORECASTS/README.md` | forecast description + the stale-submission warning |
| `04_EXPERIMENTS/README.md` | why the registry was not split |
| `04_EXPERIMENTS/EXPERIMENT_CLASSIFICATION.md` | all 86 records sorted into shipped / rejected / superseded / diagnostic |
| `05_REPORTS/README.md` | report map |
| `06_BACKEND/README.md` + `.gitkeep` | **workspace scaffold** (not built) |
| `07_FRONTEND/README.md` + `.gitkeep` | **workspace scaffold** (not built) |
| `08_DOCUMENTATION/ORGANIZATION_AUDIT.md` | this file |
| `08_DOCUMENTATION/_integrity/` | 4 verification artefacts |
| `09_SUBMISSION/README.md` | deliverable list + why the M5 submission is excluded |
| `99_ARCHIVE/README.md` | why nothing was archived |
| `PROJECT_STRUCTURE.md` | top-level map (root) |
| `scripts/08_organization/61_integrity_manifest.py` | before/after/compare hashing tool |
| `scripts/08_organization/62_experiment_classification.py` | generates the classification index |
| `scripts/08_organization/63_verify_paths.py` | 8 non-destructive path checks |

## 4. What was COPIED (originals untouched)

| Copy | Source | Verified |
|---|---|---|
| `02_MODEL/FROZEN_CHAMPION/model_11_blend_direct_final_forecast.txt` | `models/champion/` | SHA-256 identical |
| `02_MODEL/FROZEN_CHAMPION/model_12_blend_recursive_shape_final.txt` | `models/champion/` | SHA-256 identical |
| `03_FORECASTS/final_forecast_28day_v3_diversity_blend.csv` | `predictions/final_forecast/` | SHA-256 identical |
| `09_SUBMISSION/final_forecast_28day_v3_diversity_blend.csv` | `predictions/final_forecast/` | SHA-256 identical |
| `09_SUBMISSION/*.pdf` (3) | `reports/`, `MY_RESEARCH_PAPER/` | copied with `cp -p` |
| `05_REPORTS/FINAL_RESEARCH_REPORT/` (22 files) | `MY_RESEARCH_PAPER/`, `reports/` | copied with `cp -p` |

Total added by copying: ~102 MB. Every copy used `cp`; `mv` was never invoked.

## 5. What was MOVED

**Nothing.** Zero files or directories were relocated.

## 6. What was DELETED

**Nothing.** Zero files or directories were removed.

---

## 7. What was deliberately left untouched, and why

| Left in place | Why |
|---|---|
| `pipeline/` | `config.py` derives `PROJECT_ROOT` from `parent.parent`; must stay one level below root |
| `data/raw/`, `data/processed/` | resolved from `PROJECT_ROOT`; raw data is immutable by project rule |
| `models/champion/`, `models/experiments/` | resolved from `PROJECT_ROOT`; cited by the registry |
| `predictions/**` | resolved from `PROJECT_ROOT`; `39_exp76_headroom_diagnostic.py` reads 14 validation files by name |
| `experiments/registry/`, `experiments/artifacts/` | `Experiment.load()` and `load_all()` assume a flat directory; the paper builders read named records |
| `reports/`, `docs/` | resolved from `PROJECT_ROOT`; written by ~12 report scripts |
| `scripts/` | *could* move (the walk-up idiom tolerates depth), but there is no benefit and non-zero risk, so it stays |
| `MY_RESEARCH_PAPER/` | five build scripts use `parent.parent` then `import pipeline` |
| Superseded models and forecasts | cited by experiment records; deleting them destroys the evidence those experiments ran |
| `predictions/uc11_cache/` (~300 MB) | cached champion reproductions; regenerating one costs ~10 min. Reproducible, so safe to delete later if space is needed |
| `__pycache__/` (3 dirs) | regenerated automatically; already in `.gitignore` |
| `README.md`, `PROJECT_INDEX.md` and 2 other stale READMEs | documentation drift — see §9. Left untouched because they describe protected artefacts; correcting them is the owner's call |

---

## 8. Issues found that need attention

### 8.1 ⚠ The M5-format submission is STALE

`predictions/final_forecast/submission_m5_format.csv` was built from the
**superseded** `model_07` forecast (32 features, RMSE 2.1210), **not** from the
frozen champion (RMSE 2.0929). Verified by comparing its 30,490-row evaluation
block against both forecasts:

```
submission evaluation-block == final_forecast_28day (model_07, OLD)  : True
submission evaluation-block == v3_diversity_blend (FROZEN champion)  : False
```

**Action taken:** excluded from `09_SUBMISSION/` and flagged in
`03_FORECASTS/README.md`. It was **not** regenerated, because that requires
running the models — out of scope for an organisation task.

**Action needed:** if a competition-format submission of the frozen champion is
required, regenerating it is a deliberate modelling action requiring approval.

### 8.2 ⚠ The project is NOT under version control

`git ls-files` returns **0 tracked files** for `NPN_HACKATHON/`. The repository is
rooted at `C:\Users\Rishi` (the user's home directory) with a single commit, and
the entire project shows as one untracked directory. There is **no version-control
safety net** for 1.78 GB of irreplaceable model artefacts and experiment records.

**Not fixed here** — `git init` inside the project would be a structural change
beyond an organisation task, and would need a `.gitignore` strategy for the
~1.7 GB of data and models before a first commit.

**Recommended:** initialise a repository inside `NPN_HACKATHON/`, ignoring
`data/raw/`, `data/processed/`, `models/`, `predictions/` and `__pycache__/`,
so code, docs and the experiment registry are versioned while the large binaries
stay out.

### 8.3 Documentation drift (pre-existing)

Four files still describe the **superseded** 32-feature model as the champion:

| File | Says | Should say |
|---|---|---|
| `README.md` §5, §6, §14 | champion = `model_04`, 32 features, RMSE 2.1210 / MAE 1.0319; "71 experiments" | blend, 38+32 features, RMSE 2.0929 / MAE 1.0395; 86 experiments |
| `PROJECT_INDEX.md` | "The champion model → `model_04_…txt`"; "All 71 experiments" | `model_11` + `model_12`; 86 experiments |
| `models/champion/README.md` | "The selected model … RMSE 2.1210" | `model_11` + `model_12` are the shipped pair |
| `predictions/final_forecast/README.md` | quotes RMSE 2.1210 as the estimate for the delivered window; lists only 2 files | should quote 2.0929 and name the v3 blend as the deliverable |

**Left untouched by design** — these document protected artefacts, and the brief
forbids altering existing documentation to suit the new structure. The correct
values are recorded in `PROJECT_STRUCTURE.md`, `02_MODEL/MODEL_FREEZE.md` and
here. Correcting the originals is a one-line-per-file edit and is the owner's
decision.

### 8.4 Minor: `reports/` has 4 unfiled files at its root

`FINAL_MODEL_PERFORMANCE_REPORT.{md,pdf}` and
`USE_CASE_11_COMPLIANCE_AND_RESEARCH_REPORT.{md,pdf}` sit at `reports/` root
rather than in a numbered stage folder. Per `pipeline/config.py`'s own comment
("a regenerated report is written to the reports/ root and should then be filed
into the stage folder it belongs to"), filing them would match convention. **Not
done** — both are referenced by path in `09_SUBMISSION/` and
`05_REPORTS/FINAL_RESEARCH_REPORT/`, and moving them risks stale references for
purely cosmetic gain.

### 8.5 Note: `MY_RESEARCH_PAPER/` timestamps

Files in this folder carry modification times from just before the Stage 7 audit
began. They were not modified by that audit or by this reorganisation — their
hashes are recorded in `manifest_before.json` and are unchanged in
`manifest_after.json`.

---

## 9. Path issues to be aware of later

| Issue | Impact | When it matters |
|---|---|---|
| `pipeline/config.py` hard-binds the layout via `parent.parent` | The research tree cannot be reorganised without editing it | Any future restructuring |
| `MY_RESEARCH_PAPER/*.py` use `parent.parent` and `import pipeline` | That folder cannot move | If someone tries to tidy it into `05_REPORTS/` |
| `scripts/07_usecase11/{54,56,57,59}.py` import sibling `53_…` by module name | Those five files must stay in one folder | If scripts are ever split by topic |
| `Experiment.save()` never overwrites (appends `__runN`) | Re-running a script creates a new record rather than replacing one | Re-running any experiment script |
| Most pipeline scripts WRITE to `experiments/artifacts/` | Re-running them changes protected artefacts | Never re-run them casually to "test" the layout |

---

## 10. Git status

| | Before | After |
|---|---|---|
| Repository root | `C:\Users\Rishi` | unchanged |
| Branch | `main` | `main` |
| Tracked files in `NPN_HACKATHON/` | **0** | **0** |
| Project appears as | one untracked directory | one untracked directory |

No commits were made and no repository was created — both would be structural
changes outside an organisation task. See §8.2.

---

## 11. Summary

The project now has a clear delivery layer for the backend/frontend phase, and
the research pipeline underneath it is provably untouched: **0 files deleted,
0 files modified, all 5 raw-data MD5s identical, all 4 frozen copies
byte-identical, 8/8 path checks passing.**

Two things genuinely need a decision: the **stale M5 submission** (§8.1) and the
**absence of version control** (§8.2).
