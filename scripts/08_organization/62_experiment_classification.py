"""
EXPERIMENT CLASSIFICATION INDEX — generated, not hand-typed.

READ-ONLY over experiments/registry/. Writes one markdown index to
04_EXPERIMENTS/EXPERIMENT_CLASSIFICATION.md.

WHY AN INDEX INSTEAD OF MOVING FILES INTO accepted/ rejected/ archive/
----------------------------------------------------------------------
The task asked for accepted / rejected / archive organisation "ONLY if this can
be done safely without changing experiment references or registry integrity".
It cannot:

  * pipeline/experiment.py resolves records as
        config.EXPERIMENTS_DIR / f"{name}.json"
    and load_all() globs EXPERIMENTS_DIR/*.json. Both break if records move into
    subfolders.
  * MY_RESEARCH_PAPER/build_paper.py and make_figures.py read specific records by
    absolute-from-root path, e.g. REG / "exp_76_architectural_diversity_blend.json".
  * experiments/EXPERIMENT_LEDGER.md and every stage report reference records by
    flat name.

So the classification is delivered as an INDEX over the flat registry, which
achieves the navigational goal with zero risk to reproducibility. The registry
itself is left exactly as it is.

    python scripts/08_organization/62_experiment_classification.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "pipeline" / "config.py").exists())
REG = ROOT / "experiments" / "registry"
OUT = ROOT / "04_EXPERIMENTS" / "EXPERIMENT_CLASSIFICATION.md"

# --- Classification, taken from experiments/EXPERIMENT_LEDGER.md and the stage
# --- reports. Every name here is checked against the registry at run time.

SHIPPED_LINEAGE = {
    "model_02_tweedie": "established the Tweedie objective over plain L2",
    "model_04_tweedie_recency_listing": "the 32-feature base still inside member A",
    "opt_04_power_1_1": "selected variance_power = 1.1 on the inner window",
    "opt_05_recursive": "the one-step recursive architecture that became member B",
    "exp_72_per_series_shape_features": "first measured the per-series shape effect",
    "exp_73_shape_feature_validation": "accepted shape features, 4/4 windows, 3/3 seeds",
    "exp_74_shape_reproduction_and_extension": "added month/dom cycles -> the 38-feature member A",
    "exp_76_architectural_diversity_blend": "accepted the direct+recursive blend, 4/4 windows",
    "exp_77_recursive_member_upgrade": "upgraded member B and selected w = 0.60 on an inner window",
    "exp_79_upgrade_seed_check": "confirmed the upgrade is seed-stable, 6/6 cells",
    "exp_78_blend_final_forecast": "retrained the shipped blend at origin d_1941 -> the deliverable",
}

REJECTED = {
    "model_05_hurdle": "two-stage hurdle, worse",
    "opt_07_hurdle_v2": "hurdle retry, worse",
    "opt_07_hurdle_v2_calibrated": "calibrated hurdle, clearly worse",
    "opt_02_v2_A_demand": "extra demand features, worse",
    "opt_02_v2_B_calendar": "calendar expansion, worse",
    "opt_02_v2_C_price": "price dynamics, worse",
    "opt_02_v2_D_interactions": "interaction encodings, worse",
    "opt_02_v2_all": "all v2 features, worse",
    "opt_03_volume_weight_cap3": "volume weighting, worse",
    "opt_03_volume_weight_cap5": "volume weighting, worse",
    "opt_03_highvol_calibration": "high-volume calibration, no effect",
    "opt_06_obj_l1": "L1 objective, much worse RMSE",
    "opt_06_obj_l2": "L2 objective, worse",
    "opt_06_obj_poisson": "Poisson objective, worse",
    "opt_08_ensemble_tweedie_l1": "tweedie+L1 ensemble, worse",
    "exp_69_pre_origin_per_series_bias_correction": "per-series bias correction, worse",
    "exp_70_variance_reduction_ensemble": "6 same-architecture models, worse",
    "exp_71_year_over_year_features": "year-over-year features, worse",
    "model_09_tweedie_power_1_5": "power 1.5 on the primary window, worse",
    "opt_04b_power_1_5_primary": "power 1.5 confirmation, worse",
    "exp_81_reconciliation_fixed_alpha": "item-level reconciliation, failed the mechanism criterion 2/4",
    "exp_82_reconciliation_adaptive_alpha": "adaptive-alpha reconciliation, failed the same criterion",
}

SUPERSEDED = {
    "model_07_final_forecast": "forecast from the 32-feature single model — superseded by exp_78",
    "exp_75_new_champion_final_forecast": "forecast from the 38-feature shape model — superseded by exp_78",
    "model_06_tuned_primary": "reproduction of the 32-feature champion",
    "opt_00_baseline_reproduce": "reproduction of the 32-feature champion",
    "ablation_abl_7_full": "reproduction of the 32-feature champion",
    "opt_06_obj_tweedie_1_1": "reproduction of the champion objective inside the objective sweep",
    "model_01_lightgbm": "first global LightGBM (L2) — the step Tweedie improved on",
    "model_03_tweedie_recency": "intermediate step between model_02 and the 32-feature champion",
}

DIAGNOSTIC_PREFIXES = (
    "model_00_", "ablation_abl_", "tune_inner_", "probe_", "opt_09_robust_",
    "model_06_window_", "model_08_", "diagnostic_", "opt_04_power_",
    "exp_80_", "exp_80b_", "exp_80c_", "exp_83_", "exp_84_",
)


def load_all():
    out = {}
    for p in sorted(REG.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return out


def metrics_of(rec):
    m = rec.get("metrics") or {}
    r = m.get("RMSE", m.get("rmse"))
    a = m.get("MAE", m.get("mae"))
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "—"
    return fmt(r), fmt(a)


def main():
    recs = load_all()
    known = set()
    rows = {"SHIPPED": [], "REJECTED": [], "SUPERSEDED": [], "DIAGNOSTIC": [],
            "UNCLASSIFIED": []}

    for name, why in SHIPPED_LINEAGE.items():
        if name in recs:
            rows["SHIPPED"].append((name, why, *metrics_of(recs[name])))
            known.add(name)
    for name, why in REJECTED.items():
        if name in recs:
            rows["REJECTED"].append((name, why, *metrics_of(recs[name])))
            known.add(name)
    for name, why in SUPERSEDED.items():
        if name in recs:
            rows["SUPERSEDED"].append((name, why, *metrics_of(recs[name])))
            known.add(name)
    for name, rec in recs.items():
        if name in known:
            continue
        why = (rec.get("feature_set_label") or rec.get("model_type") or "")[:70]
        bucket = "DIAGNOSTIC" if name.startswith(DIAGNOSTIC_PREFIXES) else "UNCLASSIFIED"
        rows[bucket].append((name, why, *metrics_of(rec)))
        known.add(name)

    L = []
    A = L.append
    A("# Experiment classification index")
    A("")
    A(f"All **{len(recs)}** experiment records in `experiments/registry/`, "
      "classified by the role each one plays in the frozen champion.")
    A("")
    A("> **The registry itself was NOT reorganised.** Records stay flat in")
    A("> `experiments/registry/` because `pipeline/experiment.py` resolves them as")
    A("> `EXPERIMENTS_DIR / f\"{name}.json\"`, `load_all()` globs that directory, and")
    A("> `MY_RESEARCH_PAPER/build_paper.py` reads specific records by name. Moving")
    A("> them into `accepted/` `rejected/` `archive/` subfolders would break all")
    A("> three. This index gives the same navigation with zero risk.")
    A("")
    A("Generated by `scripts/08_organization/62_experiment_classification.py`.")
    A("")

    sections = [
        ("SHIPPED", "Shipped lineage — these results are inside the frozen champion",
         "Each of these contributed a component, a hyperparameter or an acceptance "
         "decision to the final model."),
        ("REJECTED", "Rejected — measured and turned down",
         "Every one of these was run and lost on evidence. They are kept because "
         "knowing what does *not* work is most of what this project established."),
        ("SUPERSEDED", "Superseded — correct when produced, replaced since",
         "These are earlier champions or their exact reproductions. They are not "
         "wrong; they are simply no longer the shipped model."),
        ("DIAGNOSTIC", "Diagnostic, baseline and audit runs",
         "Baselines, ablations, inner-window tuning, robustness sweeps and the "
         "Stage 7 audits. These inform decisions rather than being accepted or "
         "rejected themselves."),
        ("UNCLASSIFIED", "Unclassified",
         "Records that did not match any rule above. If this section is not empty, "
         "the classification script needs updating."),
    ]

    for key, title, blurb in sections:
        items = sorted(rows[key])
        if not items:
            continue
        A(f"## {title}  ({len(items)})")
        A("")
        A(blurb)
        A("")
        A("| Experiment | RMSE | MAE | Role |")
        A("|---|---|---|---|")
        for name, why, r, a in items:
            A(f"| `{name}` | {r} | {a} | {why} |")
        A("")

    A("---")
    A("")
    A("## Where to look things up")
    A("")
    A("| I want… | Path |")
    A("|---|---|")
    A("| the chronological ledger | `experiments/EXPERIMENT_LEDGER.md` |")
    A("| one experiment's full record | `experiments/registry/<name>.json` |")
    A("| result tables and diagnostics | `experiments/artifacts/` |")
    A("| backtest predictions | `predictions/validation/` |")
    A("| the stage reports | `reports/` |")
    A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    total = sum(len(v) for v in rows.values())
    print(f"  classified {total} records -> {OUT.relative_to(ROOT)}")
    for k, v in rows.items():
        print(f"    {k:<14} {len(v)}")
    if rows["UNCLASSIFIED"]:
        print("\n  UNCLASSIFIED (needs a rule):")
        for n, *_ in sorted(rows["UNCLASSIFIED"]):
            print(f"      {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
