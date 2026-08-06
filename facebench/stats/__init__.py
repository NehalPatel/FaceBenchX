"""Statistical analysis helpers for FaceBench paper experiments.

Provides mean±SD summaries, confidence intervals, paired significance
tests (McNemar, Wilcoxon), and effect sizes. Designed for fold-wise
LFW analysis and multi-model comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from facebench.metrics.recognition import (
    RecognitionMetrics,
    compute_recognition_metrics,
)


@dataclass(slots=True)
class MeanSD:
    """Mean ± standard deviation summary."""

    mean: float
    std: float
    n: int

    def format(self, digits: int = 4) -> str:
        """Format as ``mean ± std``."""
        return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConfidenceInterval:
    """Confidence interval around a point estimate."""

    estimate: float
    low: float
    high: float
    level: float = 0.95
    method: str = "wilson"

    def format(self, digits: int = 4) -> str:
        """Format as ``estimate [low, high]``."""
        pct = int(round(self.level * 100))
        return (
            f"{self.estimate:.{digits}f} "
            f"({pct}% CI {self.low:.{digits}f}–{self.high:.{digits}f})"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SignificanceResult:
    """Result of a paired statistical test."""

    test: str
    statistic: float
    p_value: float
    significant_0_05: bool
    effect_size: float | None = None
    effect_size_name: str | None = None
    n: int = 0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mean_sd(values: Sequence[float]) -> MeanSD:
    """Compute mean ± sample standard deviation."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return MeanSD(mean=float("nan"), std=float("nan"), n=0)
    if arr.size == 1:
        return MeanSD(mean=float(arr[0]), std=0.0, n=1)
    return MeanSD(mean=float(arr.mean()), std=float(arr.std(ddof=1)), n=int(arr.size))


def wilson_interval(
    successes: int,
    n: int,
    *,
    level: float = 0.95,
) -> ConfidenceInterval:
    """Wilson score confidence interval for a binomial proportion."""
    if n <= 0:
        return ConfidenceInterval(
            estimate=float("nan"),
            low=float("nan"),
            high=float("nan"),
            level=level,
            method="wilson",
        )
    z = _z_score(level)
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ConfidenceInterval(
        estimate=float(p),
        low=float(max(0.0, center - half)),
        high=float(min(1.0, center + half)),
        level=level,
        method="wilson",
    )


def proportion_ci_from_confusion(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    *,
    level: float = 0.95,
) -> dict[str, ConfidenceInterval]:
    """Wilson CIs for accuracy, precision, recall from confusion counts."""
    n = tp + tn + fp + fn
    return {
        "accuracy": wilson_interval(tp + tn, n, level=level),
        "precision": wilson_interval(tp, tp + fp, level=level),
        "recall": wilson_interval(tp, tp + fn, level=level),
        "far": wilson_interval(fp, fp + tn, level=level),
        "frr": wilson_interval(fn, fn + tp, level=level),
    }


def hanley_mcneil_auc_ci(
    auc: float,
    n_positive: int,
    n_negative: int,
    *,
    level: float = 0.95,
) -> ConfidenceInterval:
    """Approximate AUC CI using the Hanley–McNeil standard error."""
    if n_positive <= 0 or n_negative <= 0:
        return ConfidenceInterval(
            estimate=auc,
            low=float("nan"),
            high=float("nan"),
            level=level,
            method="hanley_mcneil",
        )
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    se2 = (
        auc * (1.0 - auc)
        + (n_positive - 1) * (q1 - auc**2)
        + (n_negative - 1) * (q2 - auc**2)
    ) / (n_positive * n_negative)
    se = float(np.sqrt(max(se2, 0.0)))
    z = _z_score(level)
    return ConfidenceInterval(
        estimate=float(auc),
        low=float(max(0.0, auc - z * se)),
        high=float(min(1.0, auc + z * se)),
        level=level,
        method="hanley_mcneil",
    )


def bootstrap_metric_ci(
    y_true: np.ndarray | Sequence[int],
    y_score: np.ndarray | Sequence[float],
    *,
    metric: str = "auc",
    threshold: float | None = 0.4,
    level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> ConfidenceInterval:
    """Bootstrap CI for a recognition metric."""
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true_arr.size != y_score_arr.size or y_true_arr.size == 0:
        raise ValueError("y_true and y_score must be non-empty and aligned")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    n = y_true_arr.size
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        metrics = compute_recognition_metrics(
            y_true_arr[idx],
            y_score_arr[idx],
            threshold=threshold,
        )
        estimates[i] = float(getattr(metrics, metric))

    alpha = 1.0 - level
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    point = float(getattr(
        compute_recognition_metrics(y_true_arr, y_score_arr, threshold=threshold),
        metric,
    ))
    return ConfidenceInterval(
        estimate=point,
        low=float(low),
        high=float(high),
        level=level,
        method=f"bootstrap_{n_bootstrap}",
    )


def fold_metrics(
    y_true: np.ndarray | Sequence[int],
    y_score: np.ndarray | Sequence[float],
    folds: np.ndarray | Sequence[int],
    *,
    threshold: float | None = 0.4,
) -> dict[str, Any]:
    """Compute per-fold recognition metrics and mean±SD summaries."""
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_score_arr = np.asarray(y_score, dtype=np.float64).reshape(-1)
    folds_arr = np.asarray(folds).reshape(-1)
    if not (y_true_arr.size == y_score_arr.size == folds_arr.size):
        raise ValueError("y_true, y_score, and folds must have equal length")

    per_fold: list[dict[str, Any]] = []
    metric_names = ["accuracy", "precision", "recall", "f1", "auc", "eer", "far", "frr"]
    buckets: dict[str, list[float]] = {name: [] for name in metric_names}

    for fold_id in sorted(set(int(f) for f in folds_arr if f is not None)):
        mask = folds_arr == fold_id
        if int(mask.sum()) < 2:
            continue
        metrics = compute_recognition_metrics(
            y_true_arr[mask],
            y_score_arr[mask],
            threshold=threshold,
        )
        row = {"fold": fold_id, "n": int(mask.sum())}
        for name in metric_names:
            value = float(getattr(metrics, name))
            row[name] = value
            buckets[name].append(value)
        per_fold.append(row)

    summary = {name: mean_sd(values).to_dict() for name, values in buckets.items()}
    return {"per_fold": per_fold, "summary": summary, "n_folds": len(per_fold)}


def mcnemar_test(
    y_true: np.ndarray | Sequence[int],
    y_pred_a: np.ndarray | Sequence[int],
    y_pred_b: np.ndarray | Sequence[int],
    *,
    continuity_correction: bool = True,
) -> SignificanceResult:
    """McNemar's test for paired binary classifiers on the same items."""
    truth = np.asarray(y_true).astype(bool).reshape(-1)
    pred_a = np.asarray(y_pred_a).astype(bool).reshape(-1)
    pred_b = np.asarray(y_pred_b).astype(bool).reshape(-1)
    if not (truth.size == pred_a.size == pred_b.size) or truth.size == 0:
        raise ValueError("Inputs must be non-empty and aligned")

    correct_a = pred_a == truth
    correct_b = pred_b == truth
    b = int(np.logical_and(correct_a, ~correct_b).sum())  # A right, B wrong
    c = int(np.logical_and(~correct_a, correct_b).sum())  # A wrong, B right
    n_discordant = b + c
    if n_discordant == 0:
        return SignificanceResult(
            test="mcnemar",
            statistic=0.0,
            p_value=1.0,
            significant_0_05=False,
            effect_size=0.0,
            effect_size_name="cohen_g",
            n=int(truth.size),
            notes="No discordant pairs",
            extra={"b": b, "c": c},
        )

    if continuity_correction:
        stat = (abs(b - c) - 1.0) ** 2 / n_discordant
    else:
        stat = (b - c) ** 2 / n_discordant
    # chi-square df=1 survival function
    p_value = float(_chi2_sf(stat, df=1))
    # Cohen's g = |b/(b+c) - 0.5|
    cohen_g = abs(b / n_discordant - 0.5)
    return SignificanceResult(
        test="mcnemar",
        statistic=float(stat),
        p_value=p_value,
        significant_0_05=p_value < 0.05,
        effect_size=float(cohen_g),
        effect_size_name="cohen_g",
        n=int(truth.size),
        extra={"b": b, "c": c, "n_discordant": n_discordant},
    )


def wilcoxon_signed_rank(
    differences: Sequence[float],
    *,
    zero_method: str = "wilcox",
) -> SignificanceResult:
    """Wilcoxon signed-rank test on paired differences (e.g. fold metrics).

    Args:
        differences: Paired differences (model_a - model_b) per fold/unit.
        zero_method: ``wilcox`` drops zeros; ``pratt`` keeps them.
    """
    diff = np.asarray(differences, dtype=np.float64).reshape(-1)
    if zero_method == "wilcox":
        diff = diff[diff != 0]
    if diff.size == 0:
        return SignificanceResult(
            test="wilcoxon_signed_rank",
            statistic=0.0,
            p_value=1.0,
            significant_0_05=False,
            n=0,
            notes="No non-zero differences",
        )

    abs_diff = np.abs(diff)
    ranks = _rankdata(abs_diff)
    w_pos = float(ranks[diff > 0].sum())
    w_neg = float(ranks[diff < 0].sum())
    w = min(w_pos, w_neg)
    n = int(diff.size)
    # Normal approximation with continuity correction
    mean_w = n * (n + 1) / 4.0
    se_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if se_w == 0:
        p_value = 1.0
        z = 0.0
    else:
        z = (w - mean_w) / se_w
        # two-sided
        p_value = float(2.0 * _norm_sf(abs(z)))
    # Effect size r = Z / sqrt(N)
    effect = float(abs(z) / np.sqrt(n)) if n else None
    return SignificanceResult(
        test="wilcoxon_signed_rank",
        statistic=float(w),
        p_value=min(p_value, 1.0),
        significant_0_05=p_value < 0.05,
        effect_size=effect,
        effect_size_name="r",
        n=n,
        extra={"w_pos": w_pos, "w_neg": w_neg, "z": float(z)},
    )


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def cohens_d(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Cohen's d for two independent samples (fold-level metrics)."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = np.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def predictions_at_threshold(
    y_score: np.ndarray | Sequence[float],
    threshold: float,
) -> np.ndarray:
    """Binary predictions from scores at a fixed threshold."""
    return (np.asarray(y_score, dtype=np.float64).reshape(-1) >= threshold).astype(int)


def _z_score(level: float) -> float:
    # Common levels without requiring scipy
    mapping = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    if level in mapping:
        return mapping[level]
    # Fallback approximation via inverse erf
    alpha = 1.0 - level
    return float(np.sqrt(2) * _erfinv(1.0 - alpha))


def _erfinv(x: float) -> float:
    # Approximate inverse erf (Winitzki)
    a = 0.147
    ln = np.log(1 - x**2)
    first = 2 / (np.pi * a) + ln / 2
    return float(np.sign(x) * np.sqrt(np.sqrt(first**2 - ln / a) - first))


def _norm_sf(z: float) -> float:
    """Standard normal survival function P(Z > z)."""
    return float(0.5 * (1.0 - _erf(z / np.sqrt(2.0))))


def _erf(x: float) -> float:
    # Abramowitz and Stegun approximation
    sign = 1.0 if x >= 0 else -1.0
    ax = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return float(sign * y)


def _chi2_sf(x: float, df: int = 1) -> float:
    """Chi-square survival function for df=1 (equals erfc(sqrt(x/2)))."""
    if df != 1:
        raise NotImplementedError("Only df=1 is supported without scipy")
    return float(_norm_sf(np.sqrt(x)) * 2.0)  # chi2(df=1) = z^2


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for Wilcoxon (1-based)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks
