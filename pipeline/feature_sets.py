"""
Named feature-set definitions used by the model experiments.

Keeping these in one place means every experiment refers to a feature set by
name, so the ablation table can never drift from what was actually trained.

Group letters follow features.FEATURE_GROUPS:
    A calendar   B historical demand   C recency   D listing
    E price      F hierarchy           G horizon
"""

from __future__ import annotations

from .features import FEATURE_GROUPS

A = FEATURE_GROUPS["A_calendar"]
B = FEATURE_GROUPS["B_historical_demand"]
C = FEATURE_GROUPS["C_recency"]
D = FEATURE_GROUPS["D_listing"]
E = FEATURE_GROUPS["E_price"]
F = FEATURE_GROUPS["F_hierarchy"]
G = FEATURE_GROUPS["G_horizon"]


# --------------------------------------------------------------------------
# Main model progression
#
# BASE deliberately excludes recency (C) and listing (D) so that Models 3 and 4
# can measure what those groups are actually worth. Model 1 vs Model 2 changes
# only the objective, holding features fixed.
# --------------------------------------------------------------------------

FEATURE_SETS: dict[str, list[str]] = {
    "base":                 A + B + E + F + G,
    "base_recency":         A + B + C + E + F + G,
    "base_recency_listing": A + B + C + D + E + F + G,   # == every feature

    # ---- feature-group ablation ladder (spec section 9) ----
    "abl_1_calendar":                 A,
    "abl_2_calendar_demand":          A + B,
    "abl_3_plus_recency":             A + B + C,
    "abl_4_plus_price":               A + B + C + E,
    "abl_5_plus_listing":             A + B + C + D + E,
    "abl_6_plus_hierarchy":           A + B + C + D + E + F,
    "abl_7_full":                     A + B + C + D + E + F + G,
}

# Human-readable description for the reports.
FEATURE_SET_LABELS: dict[str, str] = {
    "base": "Calendar + Historical demand + Price + Hierarchy + Horizon",
    "base_recency": "BASE + Recency",
    "base_recency_listing": "BASE + Recency + Listing (all 32 features)",
    "abl_1_calendar": "A. Calendar only",
    "abl_2_calendar_demand": "B. Calendar + Historical demand",
    "abl_3_plus_recency": "C. + Recency",
    "abl_4_plus_price": "D. + Price",
    "abl_5_plus_listing": "E. + Listing-aware",
    "abl_6_plus_hierarchy": "F. + Hierarchy",
    "abl_7_full": "G. Full feature set (+ horizon)",
}


def get(name: str) -> list[str]:
    if name not in FEATURE_SETS:
        raise KeyError(f"unknown feature set '{name}'. Known: {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[name])


def groups_in(name: str) -> list[str]:
    """Which group letters a named set contains."""
    cols = set(FEATURE_SETS[name])
    out = []
    for letter, grp in [("A", A), ("B", B), ("C", C), ("D", D),
                        ("E", E), ("F", F), ("G", G)]:
        if cols & set(grp):
            out.append(letter)
    return out
