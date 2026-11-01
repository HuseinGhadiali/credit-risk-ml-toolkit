import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import credit as C  # noqa: E402


def _toy(n=500, seed=0):
    rng = np.random.default_rng(seed)
    d = {"LIMIT_BAL": rng.integers(10_000, 500_000, n).astype(float), "SEX": rng.integers(1, 3, n),
         "EDUCATION": rng.integers(0, 7, n), "MARRIAGE": rng.integers(0, 4, n), "AGE": rng.integers(21, 70, n)}
    for i in range(1, 7):
        d[f"PAY_{i}"] = rng.integers(-2, 4, n)
        d[f"BILL_AMT{i}"] = rng.integers(0, 200_000, n).astype(float)
        d[f"PAY_AMT{i}"] = rng.integers(0, 20_000, n).astype(float)
    d[C.TARGET] = rng.integers(0, 2, n)
    return pd.DataFrame(d)


def test_clean_maps_undocumented_codes():
    d = C.clean(_toy())
    assert set(d.EDUCATION) <= {1, 2, 3, 4} and set(d.MARRIAGE) <= {1, 2, 3}


def test_engineered_features_are_finite():
    d = C.engineer(C.clean(_toy()))
    new = ["months_delinquent", "max_delay_months", "utilisation_latest", "payment_ratio_avg", "payment_ratio_min", "months_no_payment"]
    assert np.isfinite(d[new].to_numpy()).all()
    assert d.months_delinquent.between(0, 6).all() and d.payment_ratio_avg.between(0, 2).all()


def test_gini_ks_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert C.gini(y, np.r_[np.zeros(50), np.ones(50)]) == pytest.approx(1.0)
    assert C.ks_stat(y, np.r_[np.zeros(50), np.ones(50)]) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert abs(C.gini(np.tile(y, 50), rng.random(5000))) < 0.06


def test_score_scaling_properties():
    s = C.pd_to_score([0.1, 0.5, 1 / 3])
    assert s[0] == 650                                       # base odds 9:1 -> base score
    assert C.pd_to_score([1 / 19])[0] - C.pd_to_score([0.1])[0] == pytest.approx(50, abs=1)   # odds 9 -> 18 = +50 points
    assert C.pd_to_score([0.9999])[0] >= 300 and C.pd_to_score([1e-9])[0] <= 900


def test_psi():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 20_000)
    assert C.psi(a, rng.normal(0, 1, 20_000)) < 0.01
    assert C.psi(a, rng.normal(1, 1, 20_000)) > 0.25


def test_gains_table_and_cutoff():
    rng = np.random.default_rng(2)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    g = C.gains_table(y, p)
    assert g.cum_defaulters_captured_pct.iloc[-1] == pytest.approx(100)
    assert g.bad_rate_pct.iloc[0] > g.bad_rate_pct.iloc[-1]
    cut = C.cutoff_strategy(y, p)
    assert cut.bad_rate_pct.is_monotonic_increasing


def test_reason_codes_only_positive_contributions():
    sv = np.array([[0.5, -0.2, 0.1], [-0.1, -0.3, -0.2]])
    r = C.reason_codes(sv, ["PAY_1", "LIMIT_BAL", "AGE"])
    assert r[0] == ["Payment delay last month", "Age"] and r[1] == []
