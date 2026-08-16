# Independent Technical Assessment — NPN_HACKATHON

Companion to `MY_RESEARCH_PAPER.pdf`. Every figure below was re-derived from project artifacts during the audit; none is taken on trust.

## Ratings

| Dimension | Rating |
|---|---|
| Overall model quality | **Moderate** |
| Validation quality | **Good** |
| Leakage status | **Clean (independently verified)** |
| Reproducibility | **Good** |

*Moderate*, not *Good*, on model quality: the gains are real and replicated, but small in absolute terms on a noisy target, and the shipped configuration regresses MAE.

## Final model

```
y_hat = 0.60 * Direct(38 features)  +  0.40 * Recursive(32 features)

  Direct    : LightGBM Tweedie(p=1.1), 400 rounds, 15 origins x 28 days
  Recursive : LightGBM Tweedie(p=1.1), 400 rounds, 420 daily origins,
              rolled forward 28 days on its own output
  Weight    : selected on inner window d_1886-d_1913 (pre-origin)
```

## Verified performance — primary window, 853,720 predictions

| Metric | Value |
|---|---|
| RMSE | **2.0929** |
| MAE | **1.0395** |
| WAPE | 0.7205 |
| Bias | -0.0224 |
| High-volume RMSE | 5.8662 |
| Demand-occurrence accuracy (y>0, thr 0.5) | 0.6980 |
| Precision / Recall / F1 | 0.6321 / 0.8068 / 0.7088 |

Independently reproduced from raw CSVs: RMSE 2.0929 vs recorded 2.0929 (drift 3.9e-05).

Across four disjoint windows: mean ΔRMSE **-0.0242**, mean ΔMAE **+0.0186** versus the direct member alone.

## Leakage status

| Test | Result |
|---|---|
| Future sales corrupted → features change? | **0 of 38** → PASS |
| Future prices corrupted → price features change? | Yes → PASS (mirror test) |
| Recursive rollout reads post-origin actuals? | No — structurally impossible, verified |
| Train/validation target overlap | None (train ends d_1913, validation starts d_1914) |
| Target encoding / global normalisation | Not used |
| Blend weight or hyperparameters fitted on evaluation data | No |

**Conclusion: no target leakage.** Safety is structural — lags are origin-relative by construction — and empirically verified, not asserted.

## Defects found

1. **Mislabelled prediction file.** `predictions/validation/exp_74_new_champion_validation.csv` is byte-identical to `exp_72_shape_validation.csv` (36-feature model), not the 38-feature champion. Registry metrics unaffected; no downstream result depends on it.
2. **Registry does not name the shipped configuration.** Exp. #77's `metrics` field holds the w=0.50 acceptance-test blend (2.0915/1.0433), not the shipped w=0.60 (2.0929/1.0395, in `operating_point`).
3. **Seed-convention sensitivity.** Setting `bagging_seed`/`feature_fraction_seed` explicitly vs deriving them from `seed` shifts RMSE by up to 0.005 — larger than some accepted effects. Both conventions appear in the codebase.
4. **No predictions persisted for accepted models** (the shipped model had none until this audit regenerated them).

None of these invalidates the reported performance.

## Strongest contribution

The negative-control attribution of ensemble gain: -0.0044 from averaging a reseeded copy (residual corr 0.9940) versus -0.0291 from a different architecture (corr 0.9496) — **-0.0247 attributable to architecture**. This rescued a direction the project had already rejected, and diagnosed *why* the earlier attempt failed.

## Biggest weakness

The MAE regression (+0.0186). The operating point was chosen to optimise RMSE without an explicit business loss function justifying that trade. Close behind: the final accepted gain is carried by two of four windows, and on one window the upgraded member was worse than the one it replaced.

## What to improve next

1. State the loss function; re-select the blend weight against it.
2. Persist predictions for every accepted model; fix the mislabelled file.
3. Standardise the seed convention across scripts.
4. If the business can re-forecast weekly, do that before any further modelling — it beats every remaining modelling idea.

## Evidence trail

| Artifact | Path |
|---|---|
| Audit reproduction + leakage test | `audit_reproduce.py`, `audit_verification.json` |
| Regenerated shipped predictions | `reproduction/shipped_blend_w060_validation.csv` |
| Verified metric table | `MODEL_COMPARISON.csv` |
| Figures + underlying tables | `figures/` |
| Experiment registry | `experiments/registry/` (79 records) |
