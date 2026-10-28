from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from groundedgen import metrics


# ---- classification -------------------------------------------------------

def test_accuracy_and_macro_f1_hand_computed():
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", "c"]
    assert metrics.accuracy(gold, pred) == pytest.approx(0.5)
    # a: 2/3, b: 1/2; "c" is not in gold so it is not averaged.
    assert metrics.macro_f1(gold, pred) == pytest.approx((2 / 3 + 0.5) / 2)


# ---- bootstrap --------------------------------------------------------------

def _pairs_frame(n_items: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        for attr in ("category", "color", "material"):
            rows.append({"item_id": f"fpi_{i}", "attribute": attr,
                         "halluc": bool(rng.random() < 0.3)})
    return pd.DataFrame(rows)


def test_bootstrap_ratio_contains_point_estimate():
    df = _pairs_frame()
    res = metrics.bootstrap_ratio(df["halluc"], np.ones(len(df)), df["item_id"])
    assert res.estimate == pytest.approx(df["halluc"].mean())
    assert res.low <= res.estimate <= res.high
    assert res.low < res.high


def test_cluster_bootstrap_matches_ratio_bootstrap_estimate():
    df = _pairs_frame()
    res = metrics.cluster_bootstrap(df, lambda d: d["halluc"].mean(), n_resamples=200)
    assert res.estimate == pytest.approx(df["halluc"].mean())
    assert res.low <= res.estimate <= res.high


def test_bootstrap_is_seeded():
    df = _pairs_frame()
    args = (df["halluc"], np.ones(len(df)), df["item_id"])
    assert metrics.bootstrap_ratio(*args, seed=42) == metrics.bootstrap_ratio(*args, seed=42)


def test_bootstrap_ratio_zero_denominator_is_nan():
    res = metrics.bootstrap_ratio([0, 0], [0, 0], ["a", "b"], n_resamples=50)
    assert np.isnan(res.estimate)


def test_bootstrap_ratio_difference():
    df = _pairs_frame()
    a = (df["halluc"], np.ones(len(df)), df["item_id"])
    b = (np.zeros(len(df)), np.ones(len(df)), df["item_id"])
    res = metrics.bootstrap_ratio_difference(a, b, n_resamples=500)
    assert res.estimate == pytest.approx(df["halluc"].mean())
    assert res.low <= res.estimate <= res.high
    same = metrics.bootstrap_ratio_difference(a, a, n_resamples=200)
    assert same.estimate == same.low == same.high == 0.0


# ---- risk–coverage ----------------------------------------------------------

def test_risk_coverage_perfectly_ranked():
    conf = [0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [True, True, True, False, False]
    rc = metrics.risk_coverage(conf, correct)
    assert np.allclose(rc.coverage, [0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(rc.risk, [0, 0, 0, 0.25, 0.4])
    assert rc.aurc == pytest.approx((0.25 + 0.4) / 5)


def test_risk_coverage_worst_ranking_has_higher_aurc():
    correct = [True, True, True, False, False]
    best = metrics.risk_coverage([0.9, 0.8, 0.7, 0.6, 0.5], correct)
    worst = metrics.risk_coverage([0.5, 0.6, 0.7, 0.8, 0.9], correct)
    assert worst.aurc > best.aurc


def test_risk_coverage_ties_are_grouped():
    rc = metrics.risk_coverage([0.9, 0.9, 0.5], [True, False, True])
    assert np.allclose(rc.coverage, [2 / 3, 1.0])
    assert np.allclose(rc.risk, [0.5, 1 / 3])
    assert rc.aurc == pytest.approx((0.5 * 2 + 1 / 3) / 3)


# ---- tests and corrections --------------------------------------------------

def test_mcnemar_known_values():
    x = [True] * 5 + [False] * 5
    y = [False] * 10
    res = metrics.mcnemar_exact(x, y)
    assert (res["n10"], res["n01"]) == (5, 0)
    assert res["p_value"] == pytest.approx(2 * 0.5 ** 5)
    assert res["rate_diff"] == pytest.approx(0.5)

    balanced = metrics.mcnemar_exact([True, False], [False, True])
    assert balanced["p_value"] == pytest.approx(1.0)
    assert metrics.mcnemar_exact([True, True], [True, True])["p_value"] == 1.0


def test_holm_known_values():
    adjusted, reject = metrics.holm([0.01, 0.04, 0.03], alpha=0.05)
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
    assert reject.tolist() == [True, False, False]


def test_holm_caps_at_one():
    adjusted, _ = metrics.holm([0.5, 0.9])
    assert np.all(adjusted <= 1.0)


def test_wilcoxon():
    x = [5, 4, 5, 4, 5, 5, 4, 5]
    y = [3, 2, 3, 3, 2, 3, 2, 3]
    res = metrics.wilcoxon_signed_rank(x, y)
    assert res["p_value"] < 0.05
    assert res["mean_diff"] > 0
    assert metrics.wilcoxon_signed_rank([3, 3], [3, 3])["p_value"] == 1.0


# ---- agreement --------------------------------------------------------------

def test_cohen_kappa_known_case():
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "y", "y"]
    # po = 0.75; pe = 0.5*0.25 + 0.5*0.75 = 0.5; kappa = 0.5
    assert metrics.cohen_kappa(a, b) == pytest.approx(0.5)
    assert metrics.raw_agreement(a, b) == pytest.approx(0.75)
    assert metrics.cohen_kappa(a, a) == pytest.approx(1.0)


def test_cohen_kappa_undefined_is_nan():
    assert np.isnan(metrics.cohen_kappa(["u", "u"], ["u", "u"]))


def test_krippendorff_alpha_perfect_agreement():
    pytest.importorskip("krippendorff")
    data = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    assert metrics.krippendorff_alpha(data) == pytest.approx(1.0)
