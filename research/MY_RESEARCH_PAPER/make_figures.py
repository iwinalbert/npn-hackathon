"""
Figures for my research paper.

Every figure is built from an artifact already on disk or from the reproduction
run in audit_reproduce.py. Nothing is simulated, smoothed or illustrative.

Palette: slots 1-4 of the reference categorical theme, validated for light mode
(lightness band, chroma floor, CVD separation, normal-vision floor all PASS;
the sub-3:1 contrast WARN is discharged by direct labels and by the fact that
every figure is accompanied by its numeric table in the paper).

    python MY_RESEARCH_PAPER/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import config, metrics

OUT = Path(__file__).resolve().parent / "figures"
REPRO = Path(__file__).resolve().parent / "reproduction"
ART = config.ARTIFACTS_DIR

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e5e4df"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})


def finish(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    if title:
        ax.set_title(title, loc="left", color=INK, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis=grid_axis, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {name}")


# ----------------------------------------------------------------------
def fig1_demand_distribution():
    """Job: show the target's shape — zero-inflation is the defining property."""
    from pipeline.data_loader import M5Data
    d = M5Data(load_prices=False)
    S = d.sales_wide
    vals, counts = np.unique(np.clip(S, 0, 11), return_counts=True)
    share = counts / S.size * 100

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    bars = ax.bar([str(int(v)) if v < 11 else "11+" for v in vals], share,
                  color=C1, width=0.72)
    bars[0].set_color(C2)
    for i, (b, s) in enumerate(zip(bars, share)):
        if s > 1.5:
            ax.text(b.get_x() + b.get_width() / 2, s + 0.9, f"{s:.1f}%",
                    ha="center", fontsize=7.5, color=INK)
    ax.text(0.02, 0.88, "68.0% of all (series, day) cells are zero",
            transform=ax.transAxes, fontsize=8, color=C2, fontweight="bold")
    finish(ax, "Figure 1  Daily unit-sales distribution, full panel (59.2M cells)",
           "units sold on a given day", "share of all cells (%)")
    save(fig, "fig1_demand_distribution.png")


def fig2_model_comparison():
    """Job: magnitude comparison across the model lineage."""
    T = pd.read_csv(Path(__file__).resolve().parent / "MODEL_COMPARISON.csv")
    T = T[(T.Class != "member") & (~T.Model.str.contains("last value"))].copy()
    T["RMSE"] = T.RMSE.astype(float)
    T = T.sort_values("RMSE", ascending=False)
    colors = [C3 if c == "FINAL SHIPPED" else
              (C2 if "champion" in str(c) else C1) for c in T.Class]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    y = np.arange(len(T))
    ax.barh(y, T.RMSE, color=colors, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(T.Model, fontsize=7.4)
    for i, v in enumerate(T.RMSE):
        ax.text(v + 0.006, i, f"{v:.4f}", va="center", fontsize=7.2, color=INK)
    ax.set_xlim(2.0, T.RMSE.max() * 1.075)
    ax.set_ylim(-0.9, len(T) - 0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C3, C2, C1)]
    ax.legend(handles, ["final shipped champion", "prior champion", "other"],
              loc="lower right", fontsize=7.4, ncols=3,
              bbox_to_anchor=(1.0, -0.16))
    finish(ax, "Figure 2  Validation RMSE by model (853,720 predictions, d_1914–d_1941)",
           "RMSE (units/day, lower is better)", None, grid_axis="x")
    save(fig, "fig2_model_comparison.png")


def fig3_error_concentration():
    """Two panels — never a dual axis."""
    D = pd.read_csv(OUT / "decile_table.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))

    a = axes[0]
    bars = a.bar(D.decile, D.sq_err_share_pct, color=C1, width=0.7)
    bars[-1].set_color(C2)
    a.text(9.0, D.sq_err_share_pct.iloc[-1] - 12, f"{D.sq_err_share_pct.iloc[-1]:.0f}%",
           ha="center", fontsize=8.5, color="white", fontweight="bold")
    a.set_xticks(D.decile)
    finish(a, "Share of total squared error", "volume decile (1 = sparsest)", "%")

    b = axes[1]
    b.bar(D.decile, D.RMSE, color=C1, width=0.7)
    b.set_xticks(D.decile)
    finish(b, "RMSE within decile", "volume decile", "RMSE")
    fig.suptitle("Figure 3  Error is concentrated in the high-volume tail "
                 "(shipped model)", x=0.005, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig3_error_concentration.png")


def fig4_horizon():
    H = pd.read_csv(OUT / "horizon_table.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.plot(H.horizon, H.direct_RMSE, color=C1, lw=1.6, label="direct (38f)")
    ax.plot(H.horizon, H.recursive_RMSE, color=C2, lw=1.6, label="recursive (32f)")
    ax.plot(H.horizon, H.blend_RMSE, color=C3, lw=2.2, label="blend w=0.60 (shipped)")
    ax.legend(loc="upper left", ncols=3, fontsize=7.6)
    ax.set_xticks([1, 7, 14, 21, 28])
    finish(ax, "Figure 4  RMSE by forecast day (h = 1…28 from a frozen origin)",
           "days ahead of the forecast origin", "RMSE")
    save(fig, "fig4_horizon_rmse.png")


def fig5_frontier():
    """Parametric RMSE/MAE curve over the blend weight."""
    r77 = json.loads((ART / "exp77_summary.json").read_text(encoding="utf-8"))
    fr = pd.DataFrame([w for w in r77["windows"]
                       if w["window"] == "primary_spring_2016"][0]["frontier"]["AB2"])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(fr.RMSE, fr.MAE, color=C1, lw=1.6, marker="o", ms=3.4, zorder=2)
    for w_, c, lab in [(1.00, C4, "w=1.00\ndirect only"),
                       (0.60, C3, "w=0.60\nSHIPPED"),
                       (0.50, C2, "w=0.50\nRMSE-optimal")]:
        row = fr.loc[(fr.w - w_).abs().idxmin()]
        ax.scatter([row.RMSE], [row.MAE], s=64, color=c, zorder=3,
                   edgecolor="white", linewidth=1.4)
        ax.annotate(lab, (row.RMSE, row.MAE), textcoords="offset points",
                    xytext=(8, -4), fontsize=7.4, color=INK)
    ax.invert_xaxis()
    finish(ax, "Figure 5  The RMSE/MAE trade-off across the blend weight "
               "(primary window)", "RMSE (better →)", "MAE (worse ↑)")
    save(fig, "fig5_weight_frontier.png")


def fig6_feature_importance():
    I = pd.read_csv(OUT / "champion_feature_importance.csv").head(15)[::-1]
    new6 = {"wday_ratio_52w", "wday_ratio_13w", "snap_lift", "weekend_lift",
            "month_ratio", "dom_ratio"}
    colors = [C2 if f in new6 else C1 for f in I.feature]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(np.arange(len(I)), I.pct, color=colors, height=0.68)
    ax.set_yticks(np.arange(len(I)))
    ax.set_yticklabels(I.feature, fontsize=7.6)
    for i, v in enumerate(I.pct):
        ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=7.2, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C2, C1)]
    ax.legend(handles, ["per-series shape feature (Exp. #72–74)", "original 32"],
              loc="lower right", fontsize=7.5)
    finish(ax, "Figure 6  Top 15 features by split gain, direct member",
           "share of total gain (%)", None, grid_axis="x")
    save(fig, "fig6_feature_importance.png")


def fig7_cross_window():
    W = pd.DataFrame(json.loads(
        (ART / "exp77_summary.json").read_text(encoding="utf-8"))["windows"])
    op = pd.DataFrame(json.loads(
        (config.EXPERIMENTS_DIR / "exp_77_recursive_member_upgrade.json")
        .read_text(encoding="utf-8"))["operating_point"])
    op = op[op.pair == "AB2"].set_index("window")
    x = np.arange(len(W))
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.bar(x - 0.19, W.A_RMSE, width=0.36, color=C1, label="direct champion (38f)")
    ax.bar(x + 0.19, [op.loc[w, "RMSE"] for w in W.window], width=0.36,
           color=C3, label="shipped blend w=0.60")
    for i, w in enumerate(W.window):
        d = op.loc[w, "RMSE"] - W.A_RMSE.iloc[i]
        ax.text(i + 0.19, op.loc[w, "RMSE"] + 0.006, f"{d:+.4f}",
                ha="center", fontsize=7.2, color=C3)
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", "\n") for w in W.window], fontsize=7.6)
    ax.set_ylim(2.0, 2.24)
    ax.legend(loc="upper left", fontsize=7.6)
    finish(ax, "Figure 7  Held-out performance across four independent 28-day windows",
           None, "RMSE")
    save(fig, "fig7_cross_window.png")


def fig8_calibration():
    P = pd.read_csv(REPRO / "shipped_blend_w060_validation.csv")
    y, p = P.y_true.to_numpy(float), P.y_pred.to_numpy(float)
    q = pd.qcut(p, 20, labels=False, duplicates="drop")
    g = pd.DataFrame({"q": q, "y": y, "p": p}).groupby("q").mean()

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    a = axes[0]
    lim = max(g.p.max(), g.y.max()) * 1.06
    a.plot([0, lim], [0, lim], color=MUTED, lw=1.0, ls="--", zorder=1)
    a.plot(g.p, g.y, color=C1, lw=1.6, marker="o", ms=3.6, zorder=2)
    a.text(0.05, 0.86, "points below the dashed line\n= over-forecasting",
           transform=a.transAxes, fontsize=7.2, color=MUTED)
    finish(a, "Calibration (20 prediction bins)", "mean predicted", "mean actual",
           grid_axis="both")

    b = axes[1]
    err = p - y
    b.hist(np.clip(err, -8, 8), bins=80, color=C1)
    b.axvline(0, color=MUTED, lw=1.0)
    b.axvline(err.mean(), color=C2, lw=1.4)
    b.text(err.mean() - 0.3, b.get_ylim()[1] * 0.82,
           f"mean {err.mean():+.3f}", fontsize=7.2, color=C2, ha="right")
    b.set_yscale("log")
    finish(b, "Residual distribution (pred − actual)", "residual (units)",
           "count (log)")
    fig.suptitle("Figure 8  Calibration and residuals, shipped model",
                 x=0.005, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig8_calibration_residuals.png")


if __name__ == "__main__":
    print("building figures...")
    fig1_demand_distribution()
    fig2_model_comparison()
    fig3_error_concentration()
    fig4_horizon()
    fig5_frontier()
    fig6_feature_importance()
    fig7_cross_window()
    fig8_calibration()
    print("done.")
