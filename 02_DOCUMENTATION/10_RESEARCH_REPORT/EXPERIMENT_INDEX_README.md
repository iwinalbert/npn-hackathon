# 04_EXPERIMENTS — index, not a relocation

The experiment record was **not moved**. It lives where the pipeline expects it:

| What | Canonical location | Count |
|---|---|---|
| Experiment records (JSON) | `experiments/registry/` | 86 |
| Result tables and diagnostics | `experiments/artifacts/` | 74 |
| Chronological ledger | `experiments/EXPERIMENT_LEDGER.md` | — |
| Backtest predictions | `predictions/validation/` | 28 |
| Experimental model files | `models/experiments/` | 11 |
| Experiment scripts | `scripts/01_foundation/` … `03_RESEARCH/scripts/08_organization/` | 58 |

## Why the registry was not split into accepted/rejected/archive

Three separate consumers resolve records by flat name:

1. `pipeline/experiment.py` — `load()` builds `EXPERIMENTS_DIR / f"{name}.json"`;
   `load_all()` globs `EXPERIMENTS_DIR/*.json`.
2. `MY_RESEARCH_PAPER/build_paper.py` and `make_figures.py` read named records
   directly, e.g. `REG / "exp_76_architectural_diversity_blend.json"`.
3. `experiments/EXPERIMENT_LEDGER.md` and every stage report cite flat names.

Moving records into subfolders breaks all three and would silently corrupt the
research paper build. The classification is therefore delivered as an index:

**→ [`EXPERIMENT_CLASSIFICATION.md`](EXPERIMENT_CLASSIFICATION.md)**

which sorts all 86 records into shipped-lineage / rejected / superseded /
diagnostic, with the metrics pulled live from the records themselves.

Regenerate it with:

```bash
python 03_RESEARCH/scripts/08_organization/62_experiment_classification.py
```
