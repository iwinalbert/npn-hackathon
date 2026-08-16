# GIT POLICY — what is versioned, what is not, and why

**Repository root:** `NPN_HACKATHON/`
**Initialised:** at the start of the product-development phase, before any
significant product code was written, so the first commit is a clean baseline of
the completed research phase.

---

## 1. The principle

> **Version the code, the provenance and the documentation.
> Never version the data, the model binaries or the prediction matrices.**

The excluded artefacts total **~1.7 GB**. They are excluded because of *size*,
not because they are unimportant — several of them are the most valuable things
in the project. Their integrity is instead protected by **SHA-256 manifests that
ARE committed** (`02_DOCUMENTATION/09_VALIDATION/_integrity/`), so a reviewer can verify that
any copy of the artefacts is bit-identical to the one the research was run on.

Nothing was moved, renamed or deleted to make this policy work.

---

## 2. What IS committed

| Category | Contents | Size |
|---|---|---|
| Research pipeline | `pipeline/` — 22 modules | ~500 KB |
| Scripts | `scripts/` — 61 run scripts across 8 stages | ~1 MB |
| **Experiment registry** | `experiments/registry/` — all **86** records | ~1 MB |
| **Experiment artefacts** | `experiments/artifacts/` — 76 result tables/diagnostics | ~1.7 MB |
| Reports | `reports/` — 27 stage reports (PDF + markdown) + 19 charts | ~3 MB |
| Documentation | `docs/` — problem statement, dataset guides, EDA, approach | ~35 MB |
| Research paper | `MY_RESEARCH_PAPER/` minus its 34 MB reproduction CSV | ~4 MB |
| Organisation layer | `01_DATA` … `99_ARCHIVE` READMEs, freeze doc, manifests | ~5 MB |
| **Integrity manifests** | `02_DOCUMENTATION/09_VALIDATION/_integrity/` — SHA-256 of 522 files | ~400 KB |
| Product code | `01_PROJECT/backend/`, `01_PROJECT/frontend/` | grows |

Total tracked: **~50 MB** — comfortably within normal repository limits.

The experiment registry and artefacts are deliberately committed in full. They
are the scientific record of 86 experiments and are small; losing them would be
far worse than carrying a few megabytes.

---

## 3. What is NOT committed, and why

| Excluded | Size | Why | How to reconstitute |
|---|---|---|---|
| `data/raw/*.csv` | 424 MB | Immutable public M5 competition inputs. Not ours to redistribute, and unchanged since day one | Public M5 dataset; MD5s in `data/raw/README.md` |
| `data/processed/*.parquet` | 287 MB | Deterministic build output of the raw CSVs | Rebuild from `data/raw/`; provenance in `data/processed/PROCESSING_REPORT.md` |
| `models/**/*.txt` | 214 MB | LightGBM boosters — large generated binaries | Retrain via the recorded scripts; **or** keep the local copies, which are hash-verified |
| `02_DOCUMENTATION/02_MODEL/FROZEN_CHAMPION/*.txt` | 29 MB | Copies of the two frozen members | SHA-256 committed in `CHAMPION_MANIFEST.json` |
| `predictions/validation/*.csv` | 600 MB | 28 backtest matrices, 853,720 rows each | Regenerate via the experiment scripts |
| `predictions/uc11_cache/` | 294 MB | Cached champion reproductions (**model cache**) | Regenerate via `pipeline/champion_blend.py` (~10 min each) |
| `predictions/final_forecast/*.csv` | 75 MB | The forecast + superseded versions | Regenerate via `41_exp77_blend_final_forecast.py` |
| `03_FORECASTS/`, `02_DOCUMENTATION/11_SUBMISSION/` CSVs | 34 MB | Copies of the above | Copies of a hash-verified source |
| `MY_RESEARCH_PAPER/reproduction/*.csv` | 34 MB | Champion validation predictions | Regenerate via `audit_reproduce.py` |
| `01_PROJECT/backend/data/*.duckdb` | varies | **Product-generated**, fully rebuildable | `make build-db` |
| `.claude/settings.local.json` | small | Local tool permissions — machine-specific, may contain local paths | n/a |
| `__pycache__/`, `node_modules/`, `.venv/` | varies | Standard generated/vendored trees | Reinstall |
| `.env`, `*.key`, `*.pem`, `credentials*` | — | **Secrets.** None currently exist in this project — the scan came back clean — but the patterns are in place before product code introduces any | n/a |

---

## 4. The trade-off this creates, stated plainly

**A fresh clone cannot run the model or the app without the local artefacts.**
That is a real cost and it is accepted deliberately:

- committing 1.7 GB would make the repository unusable for normal collaboration;
- the artefacts are not lost — they live on disk and are hash-verified;
- the *scientific record* (86 experiment records, every metric, every report) is
  fully committed, so the research remains reviewable from the repository alone;
- the product's data layer is rebuildable with one command once the artefacts are
  present.

If the frozen model binaries later need to travel with the repository, the right
mechanism is **Git LFS** for `02_DOCUMENTATION/02_MODEL/FROZEN_CHAMPION/*.txt` (29 MB, two files)
— not committing them directly. That is a deliberate future decision, not
something to do by accident.

---

## 5. Protection guarantees preserved

Initialising Git changed **nothing** about the protected artefacts:

- no file was moved, renamed, modified or deleted;
- `.gitignore` only affects what Git *tracks*, never what exists on disk;
- the integrity manifest was re-run after `git init` and confirmed
  **0 files changed, 0 deleted**.

---

## 6. Branching and commit conventions

| | |
|---|---|
| Default branch | `main` |
| Baseline commit | the completed research phase, before product implementation |
| Product work | feature branches, e.g. `feat/api-hierarchy`, merged into `main` |
| Research layer | **frozen** — commits touching `pipeline/`, `models/`, `experiments/` or `predictions/` require explicit approval |

Commit messages state *what changed and why*, and any commit that touches the
research layer must say why the freeze was lifted.
