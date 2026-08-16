# ORGANIZATION FINAL REPORT

**Retail Demand Forecasting** — reorganisation into three top-level areas
**Date:** 2026-08-16 · **Baseline commit:** `425aac5`

---

## 1. Result

```
NPN_HACKATHON/
├── 01_PROJECT/          103 files    290 MB   what we ship
├── 02_DOCUMENTATION/     55 files     84 MB   what we explain
├── 03_RESEARCH/         532 files  1,993 MB   how we got here
├── docker-compose.yml   tasks.py   Makefile   README.md
└── .gitignore  .gitattributes  .dockerignore  .env.example
```

The previous eleven top-level entries (`01_DATA` … `99_ARCHIVE`, plus nine
research directories) are gone as top-level concepts. Nothing they contained was
deleted; every file was classified by its actual role and moved.

**Verification headline: 0 research artefacts altered, 0 files lost, 149 backend
tests, 62 frontend tests, and the project's own `verify-all` all pass.**

---

## 2. What moved, and why

### `03_RESEARCH/` — moved as ONE unit, deliberately

`pipeline/`, `data/`, `models/`, `predictions/`, `experiments/`, `reports/`,
`scripts/`, `docs/`, `MY_RESEARCH_PAPER/`, `requirements.txt`.

This was the central constraint of the whole exercise. `pipeline/config.py`
computes

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

and derives `data/`, `models/`, `predictions/`, `experiments/` and `reports/` as
its **siblings**. `MY_RESEARCH_PAPER/build_paper.py` uses the same idiom, and 56
scripts plus both integrity tools walk *up* from their own location looking for
`pipeline/config.py`.

Moving them together means `PROJECT_ROOT` now resolves to `03_RESEARCH/` and
every sibling path still resolves — so **`pipeline/config.py` required no edit
at all**. Moving any one of them independently would have broken all of them.

### `01_PROJECT/` — the runnable product

| From | To |
|---|---|
| `06_BACKEND/` | `01_PROJECT/backend/` |
| `07_FRONTEND/` | `01_PROJECT/frontend/` |

### `02_DOCUMENTATION/` — eleven categories

| From | To |
|---|---|
| `README.md` (root, research-focused) | `01_PROJECT_OVERVIEW/ML_PROJECT_OVERVIEW.md` |
| `PROJECT_INDEX.md` | `01_PROJECT_OVERVIEW/PROJECT_INDEX.md` |
| *(new)* | `01_PROJECT_OVERVIEW/PROJECT_STRUCTURE.md` |
| `02_MODEL/{MODEL_FREEZE.md, README.md, FROZEN_CHAMPION/}` | `02_MODEL/` |
| `01_DATA/README.md` | `03_DATA/DATASET_OVERVIEW.md` |
| `08_DOCUMENTATION/PRODUCT_ARCHITECTURE_PLAN.md` | `04_ARCHITECTURE/` |
| `08_DOCUMENTATION/BACKEND_IMPLEMENTATION_REPORT.md` | `05_BACKEND/` |
| `08_DOCUMENTATION/FRONTEND_IMPLEMENTATION_REPORT.md` | `06_FRONTEND/` |
| `08_DOCUMENTATION/GENAI_IMPLEMENTATION_REPORT.md` | `07_GENAI/` |
| `08_DOCUMENTATION/GIT_POLICY.md` | `08_DEPLOYMENT/` |
| `08_DOCUMENTATION/ORGANIZATION_AUDIT.md` | `09_VALIDATION/ORGANIZATION_AUDIT_2026-08.md` |
| `08_DOCUMENTATION/_integrity/` | `09_VALIDATION/_integrity/` |
| `99_ARCHIVE/README.md` | `09_VALIDATION/ARCHIVE_POLICY.md` |
| `05_REPORTS/FINAL_RESEARCH_REPORT/` | `10_RESEARCH_REPORT/FINAL_RESEARCH_REPORT/` |
| `04_EXPERIMENTS/EXPERIMENT_CLASSIFICATION.md` | `10_RESEARCH_REPORT/` |
| `09_SUBMISSION/*`, `03_FORECASTS/*` | `11_SUBMISSION/` |

### What stayed at the repository root, and why

| File | Reason it cannot move |
|---|---|
| `docker-compose.yml` | Build context is the repo root. The `full` API image does `COPY 03_RESEARCH/pipeline /research/pipeline`; a context rooted at `01_PROJECT/` could not reach the research tree |
| `.dockerignore` | Only honoured at the build context root |
| `tasks.py`, `Makefile` | Single entry point; drives both the product and the research verification scripts |
| `.gitignore`, `.gitattributes`, `.env.example` | Repository-wide by definition |

---

## 3. Dependencies updated

Twenty-six files changed content. Every one was a path reference; **none was a
research artefact**.

### Functional — would have broken

| File | Change |
|---|---|
| `01_PROJECT/backend/app/config.py` | `PROJECT_ROOT` (`parents[2]`) split into `REPO_ROOT`/`BACKEND_ROOT`/`RESEARCH_ROOT`; model and forecast defaults repointed at `03_RESEARCH/` |
| `01_PROJECT/backend/scripts/build_product_db.py` | `ROOT` → `03_RESEARCH/`; champion manifest → `02_DOCUMENTATION/02_MODEL/FROZEN_CHAMPION/` |
| `01_PROJECT/backend/Dockerfile` | `COPY 06_BACKEND/…` → `01_PROJECT/backend/…`; `COPY pipeline` → `COPY 03_RESEARCH/pipeline` |
| `docker-compose.yml` | dockerfile path, frontend context, and all three `:ro` mounts |
| `tasks.py` | `BACKEND`/`FRONTEND` roots; integrity script path |
| `.gitignore` | **23 path rules rewritten** — every protected-artefact exclusion was path-specific |
| `03_RESEARCH/scripts/08_organization/61_integrity_manifest.py` | manifest output → `02_DOCUMENTATION/09_VALIDATION/_integrity/` |
| `03_RESEARCH/scripts/08_organization/62_experiment_classification.py` | output → `02_DOCUMENTATION/10_RESEARCH_REPORT/` |
| `03_RESEARCH/scripts/08_organization/63_verify_paths.py` | output path; frozen-copy comparison pairs |
| `01_PROJECT/backend/tests/test_deployment.py` | compose lives one level higher than the product root |
| `01_PROJECT/backend/tests/test_integrity.py` | champion manifest path |
| `01_PROJECT/backend/app/services/meta.py` | provenance strings the API advertises to clients |

**Three of those live under a PROTECTED root** (`03_RESEARCH/scripts/08_organization/`).
That is the one place protected files were edited. The edits are confined to
*where each tool writes its output*; no research logic, threshold or computation
was touched, and each was re-run afterwards to prove it still works.

### Documentation — 13 markdown files
Path references updated. Three documents were deliberately **not** rewritten,
because they are historical records and editing them would falsify history:
`ORGANIZATION_AUDIT_2026-08.md`, `CHAMPION_MANIFEST.json`, and everything under
`_integrity/_history_first_organization/`.

`08_DOCUMENTATION/README.md` (the old documentation index) was **deleted**: it is
entirely superseded by `PROJECT_STRUCTURE.md`, and all seven of its links were
stale. Git retains it in history.

---

## 4. Two deliberate consolidations

**One duplicate forecast removed.** `03_FORECASTS/` and `09_SUBMISSION/` each
held `final_forecast_28day_v3_diversity_blend.csv`, and both mapped to
`11_SUBMISSION/`. Before allowing the collision I verified all copies against the
baseline:

```
c099766b85b3e3f5  17,181,402  predictions/final_forecast/…   (canonical)
c099766b85b3e3f5  17,181,402  03_FORECASTS/…                 (copy)
c099766b85b3e3f5  17,181,402  09_SUBMISSION/…                (copy)
```

Byte-identical. The canonical original is untouched in
`03_RESEARCH/predictions/final_forecast/`, and one deliverable copy remains in
`11_SUBMISSION/`. 17 MB of pure duplication removed; **no content lost** — the
hash still exists at two paths, confirmed post-move.

**Historical manifests preserved, not overwritten.** The `_integrity/` manifests
key hashes by *path*, so a reorganisation invalidates them by construction.
Rather than clobber the record of the previous organisation, the four original
files were moved to `_integrity/_history_first_organization/` and a fresh
baseline was generated for the new layout.

---

## 5. Verification

Captured **before** anything moved: a content-addressed snapshot of 679 files
(~2 GB) across every research and delivery root. Content-addressing is the point
— path-keyed hashes cannot survive a move, but the *multiset of file hashes* must
be identical either side of one.

| Check | Result |
|---|---|
| Protected-artefact manifest (`61 compare`) | **522 files · 0 deleted · 0 modified · 0 added** |
| Content-addressed comparison | **0 files lost**; 26 contents changed, all deliberate path edits |
| **Research artefacts altered** (`data/ models/ predictions/ experiments/ reports/ docs/ pipeline/ MY_RESEARCH_PAPER/`) | **0 of 471** |
| Path resolution (`63_verify_paths.py`) | **8 / 8 passed** |
| `pipeline/config.py` constants | 16 / 16 resolve; `PROJECT_ROOT` → `03_RESEARCH` |
| Pipeline modules import | 21 / 21 |
| Walk-up root resolution | 8 / 8 script folders |
| `Experiment.load_all()` | **86 records**, `exp_78` found |
| Frozen copies vs canonical sources | 3 pairs byte-identical |
| Backend tests | **149 passed**, 0 skipped |
| Frontend tests | **62 passed** |
| TypeScript | clean |
| Production build | succeeds, ~2.0 s |
| GenAI live suite | 2 passed, 4 skipped (daily API quota) |
| Slow suite — **live model reproduces the frozen forecast** | **2 passed** |
| `python tasks.py verify-all` | **ALL CHECKS PASSED** |
| Docker: compose contexts, dockerfiles, mounts | 7 / 7 resolve |
| Docker: backend `COPY` sources | 0 missing |
| Broken relative links in documentation | **0** |
| Secrets: `.env` tracked | 0 · ignored ✅ |
| Secrets: key-shaped strings in worktree | **0** |
| Secrets: key-shaped strings in git history | **0** |

### The freeze guard caught a real drift, and that is the point

The first `verify-all` **failed**:

```
CHANGED (rewritten): 1
    ~ scripts/08_organization/62_experiment_classification.py
-> FAIL: PROTECTED ARTEFACTS CHANGED
```

Correct behaviour: I had re-baselined the manifest, then edited that script's
output path. The manifest noticed. I completed all edits, re-baselined, and the
run then passed cleanly. Recording it here because a verification suite that has
never failed has not been shown to work.

### Not verified

**Docker images were not built** — Docker is not installed on this machine. Every
path a build would use was checked statically (contexts, dockerfile locations,
volume sources, and every `COPY` source), but `docker compose up --build` remains
unrun. That limitation predates this reorganisation and is unchanged by it.

---

## 6. The frozen model — unchanged

```
ŷ = 0.60 × Direct LightGBM Tweedie(1.1, 38 features)
  + 0.40 × Recursive LightGBM Tweedie(1.1, 32 features)
```

Nothing was retrained, recalibrated or regenerated. No forecast was recomputed.
The `slow` suite re-executes the frozen boosters and confirms they still
reproduce the shipped forecast bit-for-bit from the relocated tree.

| Metric | Value |
|---|---|
| RMSE | 2.0929 |
| MAE | 1.0395 |
| WAPE | 0.7205 |
| Bias | −0.0224 |

**Correction to the brief.** The occurrence metrics supplied for this task were
transposed — the quoted "Accuracy 0.8068" is the *recall*, and "Precision 0.7088"
is the *F1*. Computed from the backtest artefact under the research's documented
0.5-unit rule:

| Metric | As briefed | **Verified** |
|---|---|---|
| Accuracy | 0.8068 | **0.6980** |
| Precision | 0.7088 | **0.6321** |
| Recall | 0.8076 | **0.8068** |
| F1 | 0.7082 | **0.7088** |

This is the same correction recorded in the GenAI implementation report; it is
repeated here so a reader of this document alone is not misled. RMSE, MAE, WAPE
and bias were all stated correctly.

---

## 7. What a reader should now be able to do

| Goal | Path |
|---|---|
| Run the product | root → `python tasks.py api` / `ui` |
| Deploy it | root → `docker compose up --build` |
| Understand the design | `02_DOCUMENTATION/04_ARCHITECTURE/` |
| Confirm the model is frozen | `02_DOCUMENTATION/02_MODEL/MODEL_FREEZE.md` |
| See what was rejected and why | `02_DOCUMENTATION/10_RESEARCH_REPORT/EXPERIMENT_CLASSIFICATION.md` |
| Re-verify integrity | `python tasks.py verify-integrity` |
| Read the research history | `03_RESEARCH/` |
