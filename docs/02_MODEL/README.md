# 02_MODEL — the frozen champion

**→ Read [`MODEL_FREEZE.md`](MODEL_FREEZE.md) first.**

```
FINAL CHAMPION
0.60 x Direct (38 features)  +  0.40 x Recursive (32 features)
RMSE 2.0929    MAE 1.0395
STATUS: FROZEN
```

## What is in here

| Path | Contents |
|---|---|
| `MODEL_FREEZE.md` | architecture, hyperparameters, metrics, leakage status, validation protocol, limitations, change control |
| `FROZEN_CHAMPION/model_11_blend_direct_final_forecast.txt` | member A, byte-identical copy |
| `FROZEN_CHAMPION/model_12_blend_recursive_shape_final.txt` | member B, byte-identical copy |
| `FROZEN_CHAMPION/CHAMPION_MANIFEST.json` | SHA-256 of both members + the forecast, and provenance |

Both copies were verified SHA-256-identical to their canonical sources at freeze
time.

## Canonical sources (unchanged, still what the pipeline reads)

```
models/champion/model_11_blend_direct_final_forecast.txt
models/champion/model_12_blend_recursive_shape_final.txt
```

`models/champion/` also holds three **superseded** champions — `model_04`,
`model_07`, `model_10`. They were left in place because the experiment registry
cites them. Do not confuse them with the frozen pair above; only `model_11` +
`model_12` are the shipped model.

## For the backend phase

The backend should **not** load these LightGBM boosters to serve predictions.
The 28-day forecast is already computed for every series and lives in
`docs/11_SUBMISSION/`. Serving that CSV is instant; running the recursive member is a
28-step rollout that takes minutes.

Load the boosters only if you are building a genuine re-forecasting feature, and
if you do, treat them as read-only — see the change-control section of
`MODEL_FREEZE.md`.
