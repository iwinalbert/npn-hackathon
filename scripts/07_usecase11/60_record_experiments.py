"""
Register this research branch in experiments/registry/, in the project's own
format, from the artefact JSONs the runs produced.

BOOKKEEPING ONLY. Trains nothing, evaluates nothing, and never rewrites an
existing record — Experiment.save() appends a __runN suffix rather than
overwriting, so the historical record can only grow.

Records written:
    exp_80_item_level_reconciliation_probe   inner-window go/no-go
    exp_80b_hierarchy_level_sweep            level, objective and alpha selection
    exp_80c_level_vs_crossstore              the negative control that fired
    exp_81_reconciliation_fixed_alpha        four-window validation, fixed alpha
    exp_82_reconciliation_adaptive_alpha     four-window validation, per-origin alpha
    exp_83_covariate_audit                   requirement-6 audit (no training)
    exp_84_intermittency_audit               requirement-7 audit (no training)

    python scripts/07_usecase11/60_record_experiments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "pipeline" / "config.py").exists())))

from pipeline import config
from pipeline.experiment import Experiment

A = config.ARTIFACTS_DIR


def load(name):
    p = A / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def log(*a):
    print(*a, flush=True)


def main():
    log("Registering the Use Case 11 research branch\n")
    written = []

    # ------------------------------------------------------------------
    probe = load("uc11_exp80_probe.json")
    if probe:
        e = Experiment(
            "exp_80_item_level_reconciliation_probe",
            model_type="Item-level aggregate LightGBM + top-down reconciliation",
            objective="regression (L2) at the aggregate level",
            feature_set_label="30 aggregate features (AGG_FEATURES)",
            n_features=30,
            validation_days=probe["window"]["targets"],
            validation_origin_day=probe["window"]["origin_day"],
            horizon=config.HORIZON, n_series=config.N_SERIES)
        e.note("Inner-window go/no-go for hierarchical reconciliation. The four "
               "pre-registered proceed criteria (item model beats the bottom-up "
               "sum; dRMSE <= -0.005; alpha interior; corruption test passes) "
               "were fixed before the first run.")
        e.note("A Tweedie objective, which wins at the bottom level, was clearly "
               "WRONG at the aggregate level (item RMSE 10.79 vs 8.18 for L2): "
               "the aggregate target is neither sparse nor zero-inflated.")
        e.set_metrics(**{k: v for k, v in probe["best_overall"].items()
                         if k != "model"})
        e.set(item_level=probe["item_level"], criteria=probe["criteria"],
              leakage_corruption_test=probe["leakage_corruption_test"],
              champion_reference=probe["champion_blend"],
              decision="PROCEED" if probe["proceed"] else "STOP")
        written.append(e.save())

    # ------------------------------------------------------------------
    sweep = load("uc11_exp80b_level_sweep.json")
    if sweep:
        e = Experiment(
            "exp_80b_hierarchy_level_sweep",
            model_type="Aggregate model + reconciliation, three hierarchy levels",
            objective="regression (L2) at the aggregate level",
            feature_set_label="AGG_FEATURES at store_dept / item / item_state",
            n_features=30, validation_origin_day=sweep["window"],
            horizon=config.HORIZON, n_series=config.N_SERIES)
        e.note("Selects the level, the aggregate objective and alpha on the "
               "inner window so the four-window validation inherits them as "
               "constants.")
        e.note("One of three negative controls FIRED: an oracle global rescale "
               "was worth -0.0230 against the method's -0.0160, which triggered "
               "Experiment #80c rather than a promotion.")
        e.set(per_level_best=sweep["per_level_best"],
              sequential=sweep["sequential"],
              negative_controls=sweep["negative_controls"],
              selected=sweep["selected"],
              decision="SELECTION ONLY — no promotion from this run")
        written.append(e.save())

    # ------------------------------------------------------------------
    ortho = load("uc11_exp80c_orthogonality.json")
    if ortho:
        e = Experiment(
            "exp_80c_level_vs_crossstore",
            model_type="Decomposition of the reconciliation correction",
            objective="diagnostic",
            feature_set_label="n/a", n_features=0,
            validation_origin_day=ortho["window"],
            horizon=config.HORIZON, n_series=config.N_SERIES)
        e.note("Splits the item-level correction into a global (level) component "
               "and an item-specific (cross-store) remainder, to test whether "
               "the gain is hierarchical information or calibration in disguise.")
        e.note(f"On the inner window the champion over-forecasts by "
               f"{ortho['champion_bias']:+.4f} units per row. Two thirds of the "
               f"headline gain was that level anomaly; the item-specific "
               f"remainder was worth "
               f"{ortho['results']['4_reconcile_demeaned']['RMSE'] - ortho['results']['1_champion']['RMSE']:+.4f}.")
        e.set(results=ortho["results"],
              item_specific_share_of_gain_pct=ortho["item_specific_share_of_gain_pct"],
              decision="Cross-store component is real but small; both variants "
                       "carried forward to four-window validation")
        written.append(e.save())

    # ------------------------------------------------------------------
    for tag, fname, label in [
            ("exp_81_reconciliation_fixed_alpha", "uc11_exp81_four_window.json",
             "alpha fixed at the inner-window optimum"),
            ("exp_82_reconciliation_adaptive_alpha", "uc11_exp82_adaptive_alpha.json",
             "alpha selected per origin on the preceding 28 days")]:
        d = load(fname)
        if not d:
            continue
        dec = d["decisions"]
        best = min(dec, key=lambda k: dec[k]["mean_dRMSE"])
        e = Experiment(
            tag,
            model_type="Champion blend + item-level top-down reconciliation",
            objective="tweedie(1.1) bottom level, L2 aggregate level",
            feature_set_label=label, n_features=30,
            horizon=config.HORIZON, n_series=config.N_SERIES,
            validation_days="four disjoint 28-day windows")
        e.note("The protected champion is untouched; this is a post-hoc "
               "correction applied to its output.")
        e.set(decisions=dec, windows=d["windows"],
              mechanism_wins=d["mechanism_wins"],
              best_variant=best,
              accepted=any(v["accepted"] for v in dec.values()),
              decision=("PROMOTE " + best if any(v["accepted"] for v in dec.values())
                        else "REJECT — criteria not met"))
        prim = next((w for w in d["windows"]
                     if w["window"] == "primary_spring_2016"), None)
        if prim:
            v = prim[best] if best in prim else None
            if v:
                e.set_metrics(RMSE=v["RMSE"], MAE=v["MAE"],
                              high_volume_RMSE=v["highvol_RMSE"])
        written.append(e.save())

    # ------------------------------------------------------------------
    cov = load("uc11_covariate_audit.json")
    if cov:
        e = Experiment("exp_83_covariate_audit", model_type="audit",
                       objective="diagnostic", feature_set_label="n/a",
                       n_features=0, horizon=config.HORIZON,
                       n_series=config.N_SERIES,
                       validation_days="d_1914..d_1941")
        e.note("Requirement-6 audit: availability of every external covariate at "
               "the REAL forecast origin d_1941, plus the champion's residual "
               "structure across discount depth, price age, events and SNAP.")
        e.set(availability=cov["availability"],
              promotion_field_exists=cov["promotion_field_exists"],
              decision="audit only — no model change")
        written.append(e.save())

    inter = load("uc11_intermittency_audit.json")
    if inter:
        e = Experiment("exp_84_intermittency_audit", model_type="audit",
                       objective="diagnostic", feature_set_label="n/a",
                       n_features=0, horizon=config.HORIZON,
                       n_series=config.N_SERIES,
                       validation_days="d_1914..d_1941")
        e.note("Requirement-7 audit: Syntetos-Boylan regimes, Croston / SBA / TSB "
               "measured against the champion in every regime, and the oracle "
               "for regime specialisation.")
        e.set(regimes=inter["regimes"],
              classical_methods_per_regime=inter["classical_methods_per_regime"],
              regimes_where_classical_wins=inter["regimes_where_classical_wins"],
              per_regime_rescale_oracle_gain=inter["per_regime_rescale_oracle_gain"],
              decision="audit only — no model change")
        written.append(e.save())

    log(f"\n  {len(written)} records written to experiments/registry/")
    for p in written:
        log(f"    {p.name}")


if __name__ == "__main__":
    main()
