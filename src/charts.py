"""Charts for the credit-risk notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from credit import ROOT
from viz_style import MUTED, SERIES, TEXT, TEXT_2, apply_style, finish

IMG = ROOT / "images"


def styled(fn):
    """Re-apply the house style before every chart: optuna's plotting helpers switch matplotlib to 'ggplot' globally."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        plt.style.use("default")
        apply_style()
        return fn(*args, **kwargs)
    return wrapper


apply_style()


@styled
def lazypredict_chart(lb: pd.DataFrame, path=IMG / "lazypredict_leaderboard.png", top: int = 15):
    d = lb.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = [SERIES[0] if i >= len(d) - 3 else MUTED for i in range(len(d))]
    ax.barh(d.index, d["ROC AUC"], color=colors, height=0.6)
    for y, (v, t) in enumerate(zip(d["ROC AUC"], d["Time Taken"])):
        ax.text(v + 0.002, y, f"{v:.3f}  ·  {t:.1f}s", va="center", fontsize=8.5, color=TEXT)
    ax.set_xlim(d["ROC AUC"].min() - 0.03, d["ROC AUC"].max() + 0.04)
    ax.set_xlabel("Holdout ROC AUC (default hyper-parameters)")
    ax.grid(axis="y", visible=False)
    finish(fig, ax, "LazyPredict: 25+ classifiers screened in one call",
           "Gradient boosting leads out of the box; this decides what is worth tuning", path)


@styled
def calibration_chart(curves: dict, path=IMG / "calibration.png"):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], color=TEXT_2, lw=1)
    for i, (name, (mp, fp)) in enumerate(curves.items()):
        ax.plot(mp, fp, marker="o", markersize=5, color=SERIES[i], label=name)
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper left")
    finish(fig, ax, "SMOTE and class weights distort probabilities", "Reliability curves on the holdout (diagonal = perfectly calibrated)", path)


@styled
def cutoff_chart(cut: pd.DataFrame, chosen: int, path=IMG / "cutoff_strategy.png"):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(cut.approval_rate_pct, cut.bad_rate_pct / 100, color=SERIES[0], marker="o", markersize=5)
    row = cut[cut.approval_rate_pct == chosen].iloc[0]
    ax.scatter([chosen], [row.bad_rate_pct / 100], s=120, color=SERIES[1], zorder=3)
    ax.annotate(f"Approve {chosen}% → bad rate {row.bad_rate_pct:.1f}%\n(max profit under stated assumptions)",
                (chosen, row.bad_rate_pct / 100), xytext=(-180, 30), textcoords="offset points", fontsize=9, color=TEXT,
                arrowprops={"arrowstyle": "-", "color": TEXT_2, "lw": 0.8})
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Approval rate (%) — safest applicants first")
    ax.set_ylabel("Default rate among approved")
    finish(fig, ax, "Choosing the cut-off: approval rate vs portfolio bad rate", "Holdout applicants ranked by calibrated PD", path)


@styled
def fairness_chart(mf: pd.DataFrame, path=IMG / "fairness.png"):
    d = mf.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d.approval_rate, height=0.36, color=SERIES[0], label="Approval rate")
    ax.barh(y - 0.2, d.true_positive_rate, height=0.36, color=MUTED, label="Defaulters caught (TPR)")
    ax.set_yticks(y, d.index)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    finish(fig, ax, "Fairness audit with fairlearn", "Approval and error rates by group at the chosen cut-off", path)


@styled
def score_band_chart(t: pd.DataFrame, path=IMG / "score_bands.png"):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(t.index.astype(str), t.default_rate_pct / 100, color=SERIES[0], width=0.6)
    for i, (v, s) in enumerate(zip(t.default_rate_pct, t.share_pct)):
        ax.text(i, v / 100 + 0.01, f"{v:.0f}%\n({s:.0f}% of book)", ha="center", fontsize=8.5, color=TEXT)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, t.default_rate_pct.max() / 100 * 1.25)
    ax.set_xlabel("Credit score band (300-900 scale, 50 points double the odds)")
    ax.set_ylabel("Default rate")
    finish(fig, ax, "From probability to a CIBIL-style 300-900 score", "Observed default rate by score band, holdout", path)
