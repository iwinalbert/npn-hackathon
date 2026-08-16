# MODEL FREEZE — FINAL CHAMPION

> **STATUS: FROZEN.** This model is final. It must not be retrained, re-tuned,
> re-blended, re-seeded or replaced without deliberate, explicit approval from
> the project owner. The backend/frontend phase consumes this model's output; it
> does not modify the model.

---

## 1. Final architecture

```
FINAL FORECAST = 0.60 x DIRECT(38 features) + 0.40 x RECURSIVE(32 features)
                 clipped at 0
```

Both members are global LightGBM models with a Tweedie objective, trained across
all 30,490 store-item series at once. They are not two tunings of the same model
— they are two *architecturally different* forecasters, which is the entire
reason the blend works (see §6).

### Member A — DIRECT

| Property | Value |
|---|---|
| Algorithm | LightGBM |
| Objective | `tweedie`, `tweedie_variance_power = 1.1` |
| Features | **38** (`CHAMPION_FEATURES` in `pipeline/features_v5.py`) |
| Boosting rounds | 400 |
| Learning rate | 0.05 |
| `num_leaves` | 128 |
| `min_data_in_leaf` | 100 |
| `feature_fraction` / `bagging_fraction` | 0.8 / 0.8, `bagging_freq` 1 |
| `lambda_l2` | 1.0 |
| `max_cat_threshold` | 32 |
| Seed | 42, `deterministic = True`, `force_row_wise = True` |
| Training design | 15 origins x 30,490 series x 28 days, direct 28-day |
| Frozen artefact | `02_DOCUMENTATION/02_MODEL/FROZEN_CHAMPION/model_11_blend_direct_final_forecast.txt` |
| Canonical source | `models/champion/model_11_blend_direct_final_forecast.txt` |

Feature groups: calendar (A), historical demand (B), recency (C), listing (D),
price (E), hierarchy (F), horizon (G) — the 32-feature base — plus six
per-series shape/cycle features validated in Experiments #72–#74:
`wday_ratio_52w`, `wday_ratio_13w`, `snap_lift`, `weekend_lift`, `month_ratio`,
`dom_ratio`.

### Member B — RECURSIVE

| Property | Value |
|---|---|
| Algorithm | LightGBM, one-step-ahead, rolled forward 28 times |
| Objective | `tweedie`, `tweedie_variance_power = 1.1` |
| Features | **32** (`REC_COLS_V5` in `pipeline/champion_blend.py`) |
| Boosting rounds | 400 |
| Training design | 420 daily origins, `d_1521 .. d_1940`, horizon = 1 |
| Seed | 42 |
| Frozen artefact | `02_DOCUMENTATION/02_MODEL/FROZEN_CHAMPION/model_12_blend_recursive_shape_final.txt` |
| Canonical source | `models/champion/model_12_blend_recursive_shape_final.txt` |

Recency and listing features are deliberately excluded from this member: a
fractional prediction such as 0.3 fed back into the working matrix would be
counted as "a sale happened" and would corrupt `days_since_last_sale` during the
rollout.

### Blend

| Property | Value |
|---|---|
| Weight | **w = 0.60** on the direct member, 0.40 on the recursive member |
| How w was chosen | RMSE-optimal on an **inner window** (origin `d_1885`, targets `d_1886–d_1913`), which lies strictly before the primary window's targets and before the final forecast origin. Never fitted on an evaluation window. |
| Combination | `clip(0.60 * A + 0.40 * B, 0, None)` |

---

## 2. Final validation metrics

Primary window: origin `d_1913`, targets `d_1914–d_1941`
(2016-04-25 → 2016-05-22), 30,490 series x 28 days = **853,720 predictions**,
none excluded, none weighted.

| Metric | Value |
|---|---|
| **RMSE** | **2.0929** |
| **MAE** | **1.0395** |
| WAPE | 0.7205 |
| Bias | −0.0224 |
| High-volume RMSE (top tier) | 5.8662 |

Cross-window record at the shipped weight, against the direct member alone:

| Window | Dates | RMSE | MAE | ΔRMSE vs direct |
|---|---|---|---|---|
| primary_spring_2016 | 2016-04-25 → 05-22 | 2.0929 | 1.0395 | −0.0281 |
| christmas_2015 | 2015-12-12 → 2016-01-08 | 2.1547 | 0.9446 | −0.0227 |
| summer_2015 | 2015-07-16 → 08-12 | 2.1074 | 0.9909 | −0.0187 |
| autumn_2015 | 2015-10-02 → 10-29 | 2.1496 | 1.0126 | −0.0274 |
| **Mean** | | | | **−0.0242** |

Reference points on the same window: naive rolling-mean-28 scores RMSE 2.2430;
predicting zero everywhere scores 3.9161.

**Independently reproduced.** During the Use Case 11 audit (Stage 7) all four
windows were retrained from scratch and reproduced these figures to four
decimals — 2.0929 / 2.1547 / 2.1074 / 2.1496.

---

## 3. Leakage status: VERIFIED SAFE

| Check | Result |
|---|---|
| Foundation integrity + leakage suite | 46/46 passed |
| Corruption test (champion feature builder) | All post-origin sales overwritten with 9999; every feature returns bit-identical |
| Counter-check | The *target* does change under corruption, so the test cannot pass vacuously |
| Corruption test (Stage 7 aggregate builder) | 30/30 features identical |
| Recursive member structural check | Working matrix never contains real post-origin sales; pre-origin history intact; verified on every window |
| Training-origin rule | Every training origin satisfies `origin + horizon <= validation_origin` |
| Future covariates | Only calendar, events, SNAP and `sell_price` — all genuinely published in advance. `sell_price` is present for 30,490/30,490 series across all 28 forecast days |
| Post-origin sales | Never used in any feature, normalisation, encoding, calibration, blend weight or model selection |

The blend weight and every Stage 7 constant were selected on pre-origin windows
only.

---

## 4. Validation protocol

**Fixed-origin, direct 28-day backtest.** At origin *T* the model emits all 28
days at once. Every history-derived feature is frozen at *T* and held constant
across the horizon; only calendar and price vary per day, because only those are
genuinely published ahead.

| Block | Days | Dates |
|---|---|---|
| Training | `d_1 … d_1913` | 2011-01-29 … 2016-04-24 |
| Validation | `d_1914 … d_1941` | 2016-04-25 … 2016-05-22 |
| Final forecast | `d_1942 … d_1969` | 2016-05-23 … 2016-06-19 |

Acceptance required four disjoint temporal windows and multiple seeds, with
criteria fixed *before* each run:

| Experiment | Test | Result |
|---|---|---|
| #76 | blend beats direct across 4 windows | 4/4 |
| #76 | 3 seeds, both members reseeded | 3/3 |
| #76 | negative control (same architecture, reseeded) | −0.0044 vs −0.0291; −0.0247 attributable to architecture |
| #77 | recursive member upgrade, 4 windows | 4/4, mean −0.0042 |
| #79 | seed stability, 6 (window, seed) cells | 6/6 |

---

## 5. Limitations — read before building on this

1. **No promotion data exists.** M5 contains no promotion, display, feature-ad or
   coupon field. Promotion is only ever visible through price. A perfect
   discount correction was measured to be worth −0.0002 RMSE.
2. **Stockouts are unobservable.** A recorded zero may mean "nobody wanted it" or
   "it was not on the shelf". Nothing in this system distinguishes them, and no
   feature claims to.
3. **Point forecasts only.** No prediction intervals. Inventory decisions that
   need service levels will require a separate quantile layer — that would be new
   modelling, not a change to this model.
4. **Accuracy depends entirely on aggregation level.** At store-item-day the
   accuracy (1 − WAPE) is ~28%; per item across stores ~71%; per store ~93%;
   chain-wide ~95%. Errors are largely independent and cancel on aggregation.
   **Any UI must display the figure matching the decision being made** — quoting
   the bottom-level number for a chain-level decision understates the system
   badly, and the reverse overstates it.
5. **The MAE/RMSE trade-off is deliberate.** The blend costs ~+0.019 mean MAE
   against the direct member alone while gaining −0.024 mean RMSE. RMSE is the
   project's primary metric because the business cost of demand error is convex.
6. **Newly repriced items are under-forecast** (actual 1.706 vs predicted 0.711
   for items repriced within a week). That is 0.46% of rows; perfect treatment
   is worth −0.0012 RMSE. Documented, not fixed.
7. **No ground truth exists for the delivered forecast window.** No accuracy
   figure can honestly be quoted for `d_1942–d_1969`; the validation result above
   is the only defensible estimate.

---

## 6. Why this configuration, and what was rejected

The blend works because of **architectural diversity**, not averaging. A negative
control in Experiment #76 blended two same-architecture models and recovered only
−0.0044 of the −0.0291 gain; the remaining −0.0247 came from the direct/recursive
difference. Member residuals correlate 0.949, against 0.990 for the six
mutually-similar direct models Experiment #70 had ensembled and rejected.

Closed on measured evidence across 86 experiments — **do not re-litigate these
without new information:**

| Direction | Evidence | Verdict |
|---|---|---|
| Hierarchical reconciliation above item level | true-aggregate oracle ≤ −0.0221 | dead |
| Item-level reconciliation (fixed and adaptive α) | 3/4 windows but mechanism 2/4; adaptive α moves the wrong way on the failing window | rejected |
| Cross-store / cross-item features | joint oracle −0.0055 | dead |
| Promotion / discount covariates | perfect-correction oracle −0.0002 | dead |
| Croston / SBA / TSB | lose to the champion in **every** demand regime | dead |
| Regime / volume / category segmentation | oracle −0.0008 | dead |
| Hurdle (two-stage) models | 2.1267 and 2.1241, both worse | dead |
| Global or per-series recalibration | −0.0014 global; Experiment #69 failed outright | dead |
| Year-over-year features | 2.1564, worse | dead |
| Alternative objectives (L1, L2, Poisson) | all worse than Tweedie 1.1 | dead |

---

## 7. How to reproduce this model

```bash
python scripts/01_foundation/01_foundation_check.py        # 46 integrity + leakage checks
python scripts/06_research_campaign/41_exp77_blend_final_forecast.py
```

`pipeline/champion_blend.py` reproduces the shipped configuration for any origin
and caches results under `predictions/uc11_cache/`. It was used during the Stage 7
audit and reproduced the recorded metrics exactly on all four windows.

---

## 8. Change control

Any modification to this model requires, at minimum:

1. explicit approval from the project owner;
2. a new experiment id and new artefact files — never an overwrite;
3. validation across the same four disjoint windows with criteria fixed in
   advance;
4. a leakage corruption test on any new feature builder;
5. an update to this file recording what changed and why.

**The frozen artefacts in `02_DOCUMENTATION/02_MODEL/FROZEN_CHAMPION/` are byte-verified copies of
the canonical files under `models/champion/`.** Both were confirmed identical by
SHA-256 at freeze time — see `FROZEN_CHAMPION/CHAMPION_MANIFEST.json`. The
canonical files remain the ones the pipeline reads; the copies exist so the
backend phase has a stable, clearly labelled location that cannot be confused
with experimental models.
