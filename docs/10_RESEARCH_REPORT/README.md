# 05_REPORTS

## FINAL_RESEARCH_REPORT/ — the complete research deliverable

Byte-identical copies of the finished research output, gathered in one place:

| File | What it is |
|---|---|
| `MY_RESEARCH_PAPER.pdf` / `.docx` / `.md` | the research paper, three formats |
| `MODEL_COMPARISON.csv` | the model comparison table |
| `AUDIT_SUMMARY.md` | independent technical assessment |
| `audit_verification.json` | machine-readable audit results |
| `figures/` | 8 figures + 3 supporting CSVs |
| `reproduction/shipped_blend_w060_validation.csv` | the champion's own validation predictions |
| `FINAL_MODEL_PERFORMANCE_REPORT.pdf` / `.md` | full model comparison and occurrence metrics |
| `USE_CASE_11_COMPLIANCE_AND_RESEARCH_REPORT.pdf` / `.md` | Use Case 11 compliance matrix and Stage 7 research |

## Originals were NOT moved

The sources stay where the build scripts expect them:

| Original | Why it stays |
|---|---|
| `MY_RESEARCH_PAPER/` | all five build scripts resolve `ROOT = Path(__file__).parent.parent` and then `import pipeline`. Moving the folder one level deeper makes `ROOT` point at `05_REPORTS/`, which has no `pipeline/`, and every build script fails on import |
| `reports/` | resolved as `PROJECT_ROOT / "reports"` in `pipeline/config.py`; written by ~12 report scripts |

## The 25 stage reports

Still filed by stage under `reports/`:

| Stage | Folder |
|---|---|
| Foundation, leakage methodology | `reports/01_foundation/` |
| Models 0–5, comparison, project report | `reports/02_modelling/` |
| Team benchmark investigation | `reports/03_benchmark_investigation/` |
| Optimization campaign + final results | `reports/04_optimization/` |
| Error autopsy, research campaign, segmentation | `reports/05_diagnostics_and_research/` |
| Figures used across reports | `reports/charts/` |
