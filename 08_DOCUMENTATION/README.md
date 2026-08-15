# 08_DOCUMENTATION

| File | What it is |
|---|---|
| [`PRODUCT_ARCHITECTURE_PLAN.md`](PRODUCT_ARCHITECTURE_PLAN.md) | the architecture proposed and approved before implementation |
| [`BACKEND_IMPLEMENTATION_REPORT.md`](BACKEND_IMPLEMENTATION_REPORT.md) | Phase 2 backend: data layer, API, model serving, deployment |
| [`FRONTEND_IMPLEMENTATION_REPORT.md`](FRONTEND_IMPLEMENTATION_REPORT.md) | frontend: pages, design system, integrity rules, deployment |
| [`GENAI_IMPLEMENTATION_REPORT.md`](GENAI_IMPLEMENTATION_REPORT.md) | AI forecast assistant: Gemini integration, key handling, context retrieval, guardrails |
| [`GIT_POLICY.md`](GIT_POLICY.md) | what is versioned, what is excluded, and why |
| [`ORGANIZATION_AUDIT.md`](ORGANIZATION_AUDIT.md) | what this reorganisation changed, copied and deliberately left alone, plus verification results and issues found |
| [`../PROJECT_STRUCTURE.md`](../PROJECT_STRUCTURE.md) | top-level map — lives at the project root so it is the first thing seen |
| `_integrity/manifest_before.json` | SHA-256 of 520 protected files, before |
| `_integrity/manifest_after.json` | SHA-256 of 521 protected files, after |
| `_integrity/integrity_comparison.json` | the diff: 0 deleted, 0 modified |
| `_integrity/path_verification.json` | 8 path-resolution checks, all passing |

## Regenerating the verification

```bash
python scripts/08_organization/61_integrity_manifest.py before   # or: after / compare
python scripts/08_organization/63_verify_paths.py
python scripts/08_organization/62_experiment_classification.py
```

All three are read-only over the research tree and write only into
`08_DOCUMENTATION/` or `04_EXPERIMENTS/`.

## The original project documentation

Untouched, in `docs/`:

| Folder | Contents |
|---|---|
| `docs/01_problem_statement/` | PS11 walkthrough, the other team's approach doc |
| `docs/02_dataset/` | dataset guides and references |
| `docs/03_exploratory_analysis/` | first-pass exploration: 9 charts, summary CSVs |
| `docs/04_eda/` | formal EDA: report, methodology, 26 charts, 9 stat dumps, 33 tables |
| `docs/05_approach/` | planning documents, final approach, supporting evidence |
