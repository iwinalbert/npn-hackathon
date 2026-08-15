"""
Charts for the model reports.

Deliberately plain: horizontal/vertical bars and a line, no dual axes, no pie
charts, direct value labels so the reader never has to measure a bar against an
axis. Colours follow the project's existing EDA palette conventions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

NAVY = "#1a2a4a"
ACCENT = "#2c5f8a"
LIGHT = "#7fa8c9"
GOOD = "#2e7d5b"
BAD = "#a8443c"
GREY = "#8a8a8a"

CHART_DIR = config.REPORTS_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
})


def _save(fig, name: str) -> str:
    p = CHART_DIR / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return f"charts/{name}"


def model_comparison(labels, rmse, mae, benchmark_rmse=None, name="model_comparison.png"):
    """Horizontal bars, best model highlighted, optional benchmark reference line."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.42 * len(labels) + 1.6))
    order = np.argsort(rmse)[::-1]
    lab = [labels[i] for i in order]
    ypos = np.arange(len(lab))

    for ax, vals, title in ((axes[0], np.array(rmse)[order], "RMSE (lower is better)"),
                            (axes[1], np.array(mae)[order], "MAE (lower is better)")):
        best = int(np.argmin(vals))
        cols = [ACCENT] * len(vals)
        cols[best] = GOOD
        ax.barh(ypos, vals, color=cols, height=0.68)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.012, i, f"{v:.4f}", va="center", fontsize=7.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels(lab if ax is axes[0] else [""] * len(lab), fontsize=8)
        ax.set_title(title, fontsize=9.5, color=NAVY, loc="left")
        ax.set_xlim(0, max(vals) * 1.16)

    if benchmark_rmse:
        axes[0].axvline(benchmark_rmse, color=BAD, ls="--", lw=1.2)
        axes[0].text(benchmark_rmse, len(lab) - 0.35,
                     f" team-reported {benchmark_rmse}", color=BAD, fontsize=7.2,
                     va="top")
    return _save(fig, name)


def ablation_ladder(labels, rmse, deltas, name="ablation_ladder.png"):
    """RMSE at each rung, with the change contributed by each added group."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 5.4),
                                 gridspec_kw={"height_ratios": [1.35, 1]})
    x = np.arange(len(labels))

    a1.plot(x, rmse, "-o", color=ACCENT, lw=1.8, ms=5)
    for i, v in enumerate(rmse):
        a1.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.4)
    a1.set_xticks(x)
    a1.set_xticklabels([""] * len(x))
    a1.set_ylabel("RMSE")
    a1.set_title("Validation RMSE as each feature group is added",
                 fontsize=10, color=NAVY, loc="left")
    a1.set_ylim(min(rmse) * 0.96, max(rmse) * 1.06)

    d = [0 if v is None or (isinstance(v, float) and np.isnan(v)) else v for v in deltas]
    cols = [GOOD if v < 0 else (BAD if v > 0 else GREY) for v in d]
    a2.bar(x, d, color=cols, width=0.62)
    a2.axhline(0, color="#333", lw=0.8)
    for i, v in enumerate(d):
        if v != 0:
            a2.annotate(f"{v:+.4f}", (i, v), textcoords="offset points",
                        xytext=(0, 6 if v > 0 else -12), ha="center", fontsize=7.2)
    a2.set_xticks(x)
    a2.set_xticklabels(labels, rotation=22, ha="right", fontsize=7.8)
    a2.set_ylabel("Change in RMSE")
    a2.set_title("Green = the added group helped;  red = it made things worse",
                 fontsize=9, color=NAVY, loc="left")
    return _save(fig, name)


def feature_importance(features, gain_pct, top=15, name="feature_importance.png"):
    f = list(features)[:top][::-1]
    g = list(gain_pct)[:top][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 0.32 * len(f) + 1.2))
    ax.barh(np.arange(len(f)), g, color=ACCENT, height=0.7)
    for i, v in enumerate(g):
        ax.text(v + max(g) * 0.012, i, f"{v:.1f}%", va="center", fontsize=7.4)
    ax.set_yticks(np.arange(len(f)))
    ax.set_yticklabels(f, fontsize=8)
    ax.set_xlabel("Share of total model gain (%)")
    ax.set_title(f"Top {len(f)} features the model actually relied on",
                 fontsize=10, color=NAVY, loc="left")
    ax.set_xlim(0, max(g) * 1.14)
    return _save(fig, name)


def group_errors(labels, rmse, counts, title, name):
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(labels) + 1.5))
    y = np.arange(len(labels))
    ax.barh(y, rmse, color=ACCENT, height=0.68)
    for i, (v, c) in enumerate(zip(rmse, counts)):
        ax.text(v + max(rmse) * 0.012, i, f"{v:.3f}   (n={c:,})",
                va="center", fontsize=7.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("RMSE")
    ax.set_title(title, fontsize=10, color=NAVY, loc="left")
    ax.set_xlim(0, max(rmse) * 1.30)
    return _save(fig, name)


def rmse_by_horizon(horizons, rmse, name="rmse_by_horizon.png"):
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    ax.plot(horizons, rmse, "-o", color=ACCENT, lw=1.7, ms=4)
    ax.set_xlabel("Days ahead of the forecast origin")
    ax.set_ylabel("RMSE")
    ax.set_title("Does accuracy decay further into the 28-day horizon?",
                 fontsize=10, color=NAVY, loc="left")
    ax.set_xticks([1, 7, 14, 21, 28])
    return _save(fig, name)
