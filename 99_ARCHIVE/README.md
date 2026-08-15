# 99_ARCHIVE — deliberately empty

Nothing was archived, and nothing was deleted.

## Why

The reorganisation brief forbids deleting files, and every candidate for
"archiving" turned out to be load-bearing:

| Candidate | Why it was left in place |
|---|---|
| Superseded forecasts (`final_forecast_28day.csv`, `..._v2_shape_cycle.csv`) | Referenced by experiment records `model_07_final_forecast` and `exp_75_new_champion_final_forecast`; they are the evidence those experiments happened |
| Superseded champion models (`model_04`, `model_07`, `model_10`) | Same — cited in the registry, the ledger and the reports |
| Rejected-experiment predictions in `predictions/validation/` | Read by `39_exp76_headroom_diagnostic.py`, which blends them pairwise; deleting any breaks that diagnostic |
| Older stage reports in `reports/` | Cited by `PROJECT_INDEX.md` and by later reports |
| `predictions/uc11_cache/` (8 files, ~300 MB) | Cached champion reproductions from the Stage 7 audit. Regenerating one costs ~10 minutes of training. Kept as a time-saver; safe to delete later **if** disk space is needed, since they are reproducible |
| `__pycache__/` (3 dirs) | Regenerated automatically; already covered by `.gitignore` |

If archiving is wanted later, `predictions/uc11_cache/` is the only genuinely
disposable directory, and only because it is fully reproducible from
`pipeline/champion_blend.py`.
