"""Credit-default modelling toolkit: features, model screening, imbalance handling, Optuna tuning, calibration,
credit-risk metrics (Gini / KS / lift), score scaling, SHAP reason codes, fairness and stability checks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TARGET = "default"
SEED = 42


# ------------------------------------------------------------------ data & features
def load_raw() -> pd.DataFrame:
    from datasets import load_dataset
    df = load_dataset("scikit-learn/credit-card-clients")["train"].to_pandas()
    return df.rename(columns={"default.payment.next.month": TARGET, "PAY_0": "PAY_1"}).drop(columns="ID")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix undocumented codes in the UCI data: EDUCATION 0/5/6 and MARRIAGE 0 are not in the data dictionary -> 'other'."""
    d = df.copy()
    d["EDUCATION"] = d.EDUCATION.where(d.EDUCATION.isin([1, 2, 3]), 4)
    d["MARRIAGE"] = d.MARRIAGE.where(d.MARRIAGE.isin([1, 2]), 3)
    d[TARGET] = d[TARGET].astype(int)
    return d


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features a credit analyst would build from 6 months of statements."""
    d = df.copy()
    pay = d[[f"PAY_{i}" for i in range(1, 7)]]
    bill = d[[f"BILL_AMT{i}" for i in range(1, 7)]].to_numpy(float)
    paid = d[[f"PAY_AMT{i}" for i in range(1, 7)]].to_numpy(float)
    limit = d.LIMIT_BAL.to_numpy(float)
    d["months_delinquent"] = (pay > 0).sum(1)
    d["max_delay_months"] = pay.clip(lower=0).max(1)
    d["recent_delay"] = pay.PAY_1.clip(lower=0)
    d["utilisation_latest"] = bill[:, 0] / limit
    d["utilisation_avg"] = bill.mean(1) / limit
    d["utilisation_trend"] = (bill[:, 0] - bill[:, 5]) / limit
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bill[:, 1:] > 0, paid[:, :-1] / bill[:, 1:], 1.0)   # payment in month t vs bill of month t+1
    d["payment_ratio_avg"] = np.clip(ratio, 0, 2).mean(1)
    d["payment_ratio_min"] = np.clip(ratio, 0, 2).min(1)
    d["months_no_payment"] = (paid == 0).sum(1)
    d["over_limit_months"] = (bill > limit[:, None]).sum(1)
    return d


def xy(df: pd.DataFrame):
    return df.drop(columns=TARGET), df[TARGET]


# ------------------------------------------------------------------ credit-risk metrics
def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def ks_stat(y, p) -> float:
    y = np.asarray(y)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def metric_row(name: str, y, p) -> dict:
    return {"model": name, "roc_auc": roc_auc_score(y, p), "gini": gini(y, p), "ks": ks_stat(y, p), "brier": brier_score_loss(y, p)}


def gains_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Decile table sorted from riskiest to safest: bad rate, cumulative % of defaulters captured, lift."""
    d = pd.DataFrame({"y": np.asarray(y), "p": p})
    d["decile"] = pd.qcut(d.p.rank(method="first", ascending=False), bins, labels=range(1, bins + 1))
    g = d.groupby("decile", observed=True).agg(customers=("y", "size"), defaulters=("y", "sum"), avg_pd=("p", "mean"))
    g["bad_rate_pct"] = 100 * g.defaulters / g.customers
    g["cum_defaulters_captured_pct"] = 100 * g.defaulters.cumsum() / g.defaulters.sum()
    g["lift"] = g.bad_rate_pct / (100 * d.y.mean())
    return g.round(3)


def cutoff_strategy(y, p, loss_given_default: float = 0.6, avg_exposure: float = 1.0, margin: float = 0.12) -> pd.DataFrame:
    """Approve the safest X% of applicants. Profit per approved good = annual margin (interest + fees, 12% of exposure);
    loss per approved bad = loss-given-default (60% of exposure). Both are stated assumptions; break-even marginal bad rate = 16.7%.
    Returns approval rate vs bad rate vs relative profit, to pick a cut-off."""
    order = np.argsort(p)
    ys = np.asarray(y)[order]
    rows = []
    for a in np.arange(0.2, 1.0001, 0.05):
        k = int(len(ys) * a)
        bads = ys[:k].sum()
        goods = k - bads
        rows.append({"approval_rate_pct": round(100 * a), "bad_rate_pct": 100 * bads / k,
                     "profit_index": (goods * margin - bads * loss_given_default) * avg_exposure / len(ys) * 1000})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ score scaling (CIBIL-style 300-900)
def pd_to_score(p, base_score: int = 650, base_odds: float = 9.0, pdo: int = 50, lo: int = 300, hi: int = 900) -> np.ndarray:
    """Points-to-double-odds scaling: `base_score` at good:bad odds of `base_odds`; +`pdo` points doubles the odds."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    odds = (1 - p) / p
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return np.clip(np.round(offset + factor * np.log(odds)), lo, hi).astype(int)


def score_bands(y, scores) -> pd.DataFrame:
    bands = pd.cut(scores, [299, 550, 600, 650, 700, 750, 900],
                   labels=["300-550", "551-600", "601-650", "651-700", "701-750", "751-900"])
    t = pd.DataFrame({"band": bands, "y": np.asarray(y)}).groupby("band", observed=False).agg(customers=("y", "size"), default_rate=("y", "mean"))
    t["share_pct"] = 100 * t.customers / t.customers.sum()
    t["default_rate_pct"] = 100 * t.default_rate
    return t.drop(columns="default_rate").round(2)


# ------------------------------------------------------------------ stability
def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two score distributions (<0.1 stable, 0.1-0.25 watch, >0.25 shift)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(((a - e) * np.log(a / e)).sum())


# ------------------------------------------------------------------ explainability
READABLE = {
    "PAY_1": "Payment delay last month", "recent_delay": "Payment delay last month", "max_delay_months": "Worst payment delay (6 months)",
    "months_delinquent": "Months with a late payment", "LIMIT_BAL": "Credit limit", "utilisation_latest": "Current credit utilisation",
    "utilisation_avg": "Average credit utilisation",
    "payment_ratio_avg": "Share of bill usually paid", "payment_ratio_min": "Lowest share of bill paid",
    "months_no_payment": "Months with no payment", "over_limit_months": "Months over the credit limit", "PAY_AMT1": "Last payment amount",
    "BILL_AMT1": "Latest bill amount", "AGE": "Age", "utilisation_trend": "Utilisation trend",
}


def reason_codes(shap_values: np.ndarray, feature_names: list[str], top: int = 3) -> list[list[str]]:
    """Top features pushing each applicant's risk UP: the 'principal reasons' a lender must give for a decline."""
    out = []
    for row in shap_values:
        idx = np.argsort(-row)[:top]
        out.append([READABLE.get(feature_names[i], feature_names[i]) for i in idx if row[i] > 0])
    return out
