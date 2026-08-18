
from __future__ import annotations

from .features import FEATURE_GROUPS

A = FEATURE_GROUPS["A_calendar"]
B = FEATURE_GROUPS["B_historical_demand"]
C = FEATURE_GROUPS["C_recency"]
D = FEATURE_GROUPS["D_listing"]
E = FEATURE_GROUPS["E_price"]
F = FEATURE_GROUPS["F_hierarchy"]
G = FEATURE_GROUPS["G_horizon"]


FEATURE_SETS: dict[str, list[str]] = {
    "base":                 A + B + E + F + G,
    "base_recency":         A + B + C + E + F + G,
    "base_recency_listing": A + B + C + D + E + F + G,

    "abl_1_calendar":                 A,
    "abl_2_calendar_demand":          A + B,
    "abl_3_plus_recency":             A + B + C,
    "abl_4_plus_price":               A + B + C + E,
    "abl_5_plus_listing":             A + B + C + D + E,
    "abl_6_plus_hierarchy":           A + B + C + D + E + F,
    "abl_7_full":                     A + B + C + D + E + F + G,
}

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
    cols = set(FEATURE_SETS[name])
    out = []
    for letter, grp in [("A", A), ("B", B), ("C", C), ("D", D),
                        ("E", E), ("F", F), ("G", G)]:
        if cols & set(grp):
            out.append(letter)
    return out
