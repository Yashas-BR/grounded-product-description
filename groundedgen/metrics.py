"""Statistics shared by every evaluation stage (spec §7, §11, §12).

Bootstrap CIs resample *items* as clusters: all of an item's attributes and
variants move together. Everything here is pure NumPy/SciPy and runs on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ArrayLike = Sequence[Any] | np.ndarray | pd.Series


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate with a percentile bootstrap confidence interval."""

    estimate: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high}


@dataclass(frozen=True)
class RiskCoverage:
    """Risk–coverage curve: one point per accepted-set size (ties grouped)."""

    coverage: np.ndarray
    risk: np.ndarray
    thresholds: np.ndarray
    aurc: float


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def accuracy(gold: ArrayLike, pred: ArrayLike) -> float:
    """Share of positions where ``pred`` equals ``gold`` (NaN if empty)."""
    gold, pred = np.asarray(gold, dtype=object), np.asarray(pred, dtype=object)
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    return float(np.mean(gold == pred)) if len(gold) else float("nan")


def macro_f1(gold: ArrayLike, pred: ArrayLike) -> float:
    """Macro-F1 averaged over the classes present in ``gold`` (spec §7)."""
    gold, pred = np.asarray(gold, dtype=object), np.asarray(pred, dtype=object)
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    classes = list(dict.fromkeys(gold.tolist()))
    if not classes:
        return float("nan")
    scores = []
    for cls in classes:
        tp = int(np.sum((gold == cls) & (pred == cls)))
        fp = int(np.sum((gold != cls) & (pred == cls)))
        fn = int(np.sum((gold == cls) & (pred != cls)))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def _percentile_ci(samples: np.ndarray, ci: float) -> tuple[float, float]:
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - ci) / 2.0 * 100.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def cluster_bootstrap(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    cluster_col: str = "item_id",
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap of an arbitrary ``statistic`` over item clusters.

    Each resample draws clusters (items) with replacement and keeps every row
    of a drawn cluster. Use :func:`bootstrap_ratio` for plain rates; it is
    much faster.
    """
    estimate = float(statistic(df))
    groups = list(df.groupby(cluster_col, sort=True).indices.values())
    if not groups:
        return BootstrapResult(estimate, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples)
    for b in range(n_resamples):
        picks = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in picks])
        samples[b] = statistic(df.iloc[idx])
    return BootstrapResult(estimate, *_percentile_ci(samples, ci))


def _per_cluster_sums(
    numerator: ArrayLike, denominator: ArrayLike, clusters: ArrayLike
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "num": np.asarray(numerator, dtype=float),
        "den": np.asarray(denominator, dtype=float),
        "cluster": np.asarray(clusters, dtype=object),
    })
    return frame.groupby("cluster", sort=True)[["num", "den"]].sum()


def bootstrap_ratio(
    numerator: ArrayLike,
    denominator: ArrayLike,
    clusters: ArrayLike,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap CI of ``sum(numerator) / sum(denominator)`` over clusters.

    For a rate over scored pairs pass the event indicator as ``numerator`` and
    ones (or the eligibility indicator) as ``denominator``; ``clusters`` holds
    each row's item id.
    """
    sums = _per_cluster_sums(numerator, denominator, clusters)
    num, den = sums["num"].to_numpy(), sums["den"].to_numpy()
    estimate = float(_safe_ratio(np.array([num.sum()]), np.array([den.sum()]))[0])
    if len(num) == 0:
        return BootstrapResult(estimate, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(num), size=(n_resamples, len(num)))
    samples = _safe_ratio(num[picks].sum(axis=1), den[picks].sum(axis=1))
    return BootstrapResult(estimate, *_percentile_ci(samples, ci))


def bootstrap_ratio_difference(
    a: tuple[ArrayLike, ArrayLike, ArrayLike],
    b: tuple[ArrayLike, ArrayLike, ArrayLike],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap CI of ratio(a) − ratio(b) over shared item clusters.

    ``a`` and ``b`` are ``(numerator, denominator, clusters)`` triples, e.g.
    the GD indicator for condition B and for condition C (RQ1 origin test).
    Both ratios are computed on the same resampled items.
    """
    sums_a = _per_cluster_sums(*a)
    sums_b = _per_cluster_sums(*b)
    index = sums_a.index.union(sums_b.index)
    sums_a = sums_a.reindex(index, fill_value=0.0)
    sums_b = sums_b.reindex(index, fill_value=0.0)
    na, da = sums_a["num"].to_numpy(), sums_a["den"].to_numpy()
    nb, db = sums_b["num"].to_numpy(), sums_b["den"].to_numpy()

    def diff(sel: np.ndarray | slice) -> np.ndarray:
        ra = _safe_ratio(np.atleast_2d(na[sel]).sum(axis=-1), np.atleast_2d(da[sel]).sum(axis=-1))
        rb = _safe_ratio(np.atleast_2d(nb[sel]).sum(axis=-1), np.atleast_2d(db[sel]).sum(axis=-1))
        return ra - rb

    estimate = float(diff(slice(None))[0])
    if len(index) == 0:
        return BootstrapResult(estimate, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(index), size=(n_resamples, len(index)))
    return BootstrapResult(estimate, *_percentile_ci(diff(picks), ci))


# --------------------------------------------------------------------------
# Selective prediction
# --------------------------------------------------------------------------

def risk_coverage(confidence: ArrayLike, correct: ArrayLike) -> RiskCoverage:
    """Risk–coverage curve and AURC (Geifman & El-Yaniv, 2017).

    Items are accepted in order of decreasing confidence. Items with equal
    confidence are accepted together, so each tie group contributes one curve
    point and its risk to every position it spans (DECISIONS.md, M1 #6).
    AURC is the mean risk over all n positions.

    Args:
        confidence: score per item; higher means more confident.
        correct: whether the asserted prediction is correct (for
            non-determinable items an assertion is always incorrect).
    """
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    if conf.shape != ok.shape:
        raise ValueError("confidence and correct differ in shape")
    n = conf.size
    if n == 0:
        empty = np.array([])
        return RiskCoverage(empty, empty, empty, float("nan"))
    order = np.argsort(-conf, kind="stable")
    conf, errors = conf[order], (~ok[order]).astype(float)
    # Last position of each tie group.
    ends = np.flatnonzero(np.r_[conf[1:] != conf[:-1], True])
    cum_errors = np.cumsum(errors)[ends]
    accepted = ends + 1
    risk = cum_errors / accepted
    group_sizes = np.diff(np.r_[0, accepted])
    aurc = float(np.sum(risk * group_sizes) / n)
    return RiskCoverage(accepted / n, risk, conf[ends], aurc)


# --------------------------------------------------------------------------
# Hypothesis tests
# --------------------------------------------------------------------------

def mcnemar_exact(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Exact two-sided McNemar test on paired binary indicators.

    ``x`` and ``y`` are per-pair hallucination indicators for two conditions on
    the same items and attributes. Returns the discordant counts, the p-value
    and the rate difference mean(x) − mean(y) as the effect size.
    """
    x, y = np.asarray(x, dtype=bool), np.asarray(y, dtype=bool)
    if x.shape != y.shape:
        raise ValueError("x and y must be paired (same shape)")
    n10 = int(np.sum(x & ~y))
    n01 = int(np.sum(~x & y))
    discordant = n10 + n01
    p_value = 1.0 if discordant == 0 else float(
        stats.binomtest(min(n10, n01), discordant, 0.5).pvalue
    )
    rate_diff = float(x.mean() - y.mean()) if x.size else float("nan")
    return {"n10": n10, "n01": n01, "p_value": p_value, "rate_diff": rate_diff}


def holm(p_values: ArrayLike, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm–Bonferroni step-down correction.

    Returns ``(adjusted_p, reject)`` in the input order.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p.copy(), np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    stepped = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adjusted_sorted = np.maximum.accumulate(stepped)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= alpha


def wilcoxon_signed_rank(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test on paired ratings.

    Returns statistic, p-value and the mean difference. If every difference is
    zero the test is undefined; the p-value is then reported as 1.0.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must be paired (same shape)")
    mean_diff = float(np.mean(x - y)) if x.size else float("nan")
    if x.size == 0 or np.all(x == y):
        return {"statistic": float("nan"), "p_value": 1.0, "mean_diff": mean_diff}
    result = stats.wilcoxon(x, y)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_diff": mean_diff,
    }


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------

def raw_agreement(a: ArrayLike, b: ArrayLike) -> float:
    """Share of items on which two raters gave the same label."""
    return accuracy(a, b)


def cohen_kappa(a: ArrayLike, b: ArrayLike) -> float:
    """Cohen's kappa between two raters' labels on the same items.

    NaN when expected agreement is 1 (both raters used a single identical
    label), where kappa is undefined (DECISIONS.md, M1 #5).
    """
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    if len(a) != len(b):
        raise ValueError("raters labelled different numbers of items")
    if len(a) == 0:
        return float("nan")
    labels: list[Hashable] = list(dict.fromkeys([*a.tolist(), *b.tolist()]))
    observed = float(np.mean(a == b))
    expected = float(sum(np.mean(a == lab) * np.mean(b == lab) for lab in labels))
    if np.isclose(expected, 1.0):
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def krippendorff_alpha(
    reliability_data: ArrayLike, level_of_measurement: str = "ordinal"
) -> float:
    """Krippendorff's alpha via the ``krippendorff`` package.

    Args:
        reliability_data: raters × units matrix; NaN marks a missing rating.
        level_of_measurement: ``"ordinal"`` for the 1–5 human ratings.
    """
    import krippendorff  # lazy: only needed for human evaluation

    data = np.asarray(reliability_data, dtype=float)
    return float(
        krippendorff.alpha(reliability_data=data, level_of_measurement=level_of_measurement)
    )
