"""Render assets/banner.png: 30K real card holders, credit limit vs utilisation, defaulters glowing (python assets/make_banner.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from banner import glow_scatter, make_banner  # noqa: E402
from credit import clean, engineer, load_raw  # noqa: E402

d = engineer(clean(load_raw())).sample(9000, random_state=1)
x = np.log10(d.LIMIT_BAL)
y = d.utilisation_avg.clip(-0.1, 1.2)
bad = d["default"].to_numpy() == 1
rng = np.random.default_rng(0)
jx, jy = rng.normal(0, 0.03, len(d)), rng.normal(0, 0.01, len(d))


def art(ax, acc):
    glow_scatter(ax, (x + jx)[~bad], (y + jy)[~bad], acc[1], s=2.5, alpha=0.3)
    glow_scatter(ax, (x + jx)[bad], (y + jy)[bad], "#ffc94d", s=4, alpha=0.8)
    ax.text(x.min(), -0.32, "each dot = a real card holder · credit limit (log) vs average utilisation · gold = defaulted next month",
            color="white", alpha=0.55, fontsize=9)
    ax.set_ylim(-0.38, 1.3)


make_banner(Path(__file__).parent / "banner.png", "Credit Risk ML Toolkit",
            "LazyPredict · Optuna · calibration · SHAP · fairlearn",
            ["LightGBM", "Optuna", "SHAP", "fairlearn", "Gini / KS"], art, theme="ocean",
            emoji_free_kicker="BFSI · 30K real card holders")
