"""Recognition / verification metric data structures and calculators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ConfusionCounts:
    """Binary confusion-matrix cell counts.

    Attributes:
        true_positive: Same-identity pairs accepted.
        true_negative: Different-identity pairs rejected.
        false_positive: Different-identity pairs accepted (false accepts).
        false_negative: Same-identity pairs rejected (false rejects).
    """

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def as_matrix(self) -> np.ndarray:
        """Return a ``[[TN, FP], [FN, TP]]`` matrix.

        Returns:
            2x2 integer numpy array.
        """
        return np.asarray(
            [
                [self.true_negative, self.false_positive],
                [self.false_negative, self.true_positive],
            ],
            dtype=np.int64,
        )


@dataclass(slots=True)
class RecognitionMetrics:
    """Verification metrics at a fixed threshold plus ROC summary.

    Attributes:
        threshold: Decision threshold used for hard metrics.
        accuracy: Classification accuracy.
        precision: Precision for the positive (same) class.
        recall: Recall / TPR for the positive class.
        f1: F1 score.
        far: False Accept Rate (FPR).
        frr: False Reject Rate (FNR).
        auc: Area under the ROC curve.
        eer: Equal Error Rate.
        eer_threshold: Score threshold nearest to the EER operating point.
        confusion: Confusion counts at ``threshold``.
        roc_fpr: ROC false-positive rates.
        roc_tpr: ROC true-positive rates.
        roc_thresholds: Score thresholds corresponding to ROC points.
        num_pairs: Total number of evaluated pairs.
        num_positive: Number of same-identity pairs.
        num_negative: Number of different-identity pairs.
        extra: Extensible metadata.
    """

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    auc: float
    eer: float
    eer_threshold: float
    confusion: ConfusionCounts
    roc_fpr: np.ndarray = field(repr=False)
    roc_tpr: np.ndarray = field(repr=False)
    roc_thresholds: np.ndarray = field(repr=False)
    num_pairs: int = 0
    num_positive: int = 0
    num_negative: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a JSON-friendly dictionary.

        Returns:
            Plain dictionary with lists instead of numpy arrays.
        """
        payload = asdict(self)
        payload["confusion"] = asdict(self.confusion)
        payload["confusion_matrix"] = self.confusion.as_matrix().tolist()
        payload["roc_fpr"] = self.roc_fpr.astype(float).tolist()
        payload["roc_tpr"] = self.roc_tpr.astype(float).tolist()
        payload["roc_thresholds"] = self.roc_thresholds.astype(float).tolist()
        return payload


def _as_bool_labels(y_true: np.ndarray) -> np.ndarray:
    labels = np.asarray(y_true).reshape(-1)
    if labels.dtype == np.bool_:
        return labels
    return labels.astype(np.float64) > 0.5


def confusion_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> ConfusionCounts:
    """Compute confusion counts at a similarity threshold.

    Args:
        y_true: Binary labels (1/True = same identity).
        y_score: Similarity scores (higher = more similar).
        threshold: Accept when ``score >= threshold``.

    Returns:
        :class:`ConfusionCounts`.
    """
    labels = _as_bool_labels(y_true)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("y_true and y_score must have the same length")
    predicted = scores >= threshold
    tp = int(np.logical_and(predicted, labels).sum())
    tn = int(np.logical_and(~predicted, ~labels).sum())
    fp = int(np.logical_and(predicted, ~labels).sum())
    fn = int(np.logical_and(~predicted, labels).sum())
    return ConfusionCounts(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )


def binary_rates(confusion: ConfusionCounts) -> dict[str, float]:
    """Derive accuracy/precision/recall/F1/FAR/FRR from confusion counts.

    Args:
        confusion: Confusion cell counts.

    Returns:
        Mapping of metric name → value in ``[0, 1]`` (best-effort).
    """
    tp = confusion.true_positive
    tn = confusion.true_negative
    fp = confusion.false_positive
    fn = confusion.false_negative
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "far": float(far),
        "frr": float(frr),
    }


def roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an ROC curve for verification scores.

    Args:
        y_true: Binary labels.
        y_score: Similarity scores.

    Returns:
        ``(fpr, tpr, thresholds)`` arrays sorted by descending score.
    """
    labels = _as_bool_labels(y_true)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if labels.size == 0:
        return (
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([np.inf]),
        )

    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]

    distinct = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct, labels.size - 1]

    tps = np.cumsum(labels)[threshold_idxs]
    fps = np.cumsum(~labels)[threshold_idxs]
    thresholds = scores[threshold_idxs]

    positives = max(int(labels.sum()), 1)
    negatives = max(int((~labels).sum()), 1)
    tpr = tps / positives
    fpr = fps / negatives

    # Prepend origin for proper AUC integration.
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[thresholds[0] + 1e-12 if thresholds.size else 1.0, thresholds]
    return fpr.astype(np.float64), tpr.astype(np.float64), thresholds.astype(np.float64)


def roc_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal area under the ROC curve.

    Args:
        fpr: False-positive rates.
        tpr: True-positive rates.

    Returns:
        AUC in ``[0, 1]``.
    """
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(tpr, fpr))


def equal_error_rate(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float]:
    """Estimate Equal Error Rate and its operating threshold.

    Args:
        y_true: Binary labels.
        y_score: Similarity scores.

    Returns:
        ``(eer, eer_threshold)`` where EER is the FAR/FRR meeting point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    # Find threshold where FAR (fpr) ≈ FRR (fnr).
    abs_diff = np.abs(fpr - fnr)
    idx = int(np.argmin(abs_diff))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[min(idx, thresholds.size - 1)])
    return eer, threshold


def compute_recognition_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float | None = None,
) -> RecognitionMetrics:
    """Compute full recognition metrics for a verification experiment.

    Args:
        y_true: Binary same/different labels.
        y_score: Similarity scores.
        threshold: Fixed decision threshold. When ``None``, uses the
            EER operating threshold.

    Returns:
        Populated :class:`RecognitionMetrics`.
    """
    labels = _as_bool_labels(y_true)
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    eer, eer_threshold = equal_error_rate(labels, scores)
    decision_threshold = eer_threshold if threshold is None else float(threshold)
    confusion = confusion_at_threshold(labels, scores, decision_threshold)
    rates = binary_rates(confusion)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc(fpr, tpr)
    return RecognitionMetrics(
        threshold=decision_threshold,
        accuracy=rates["accuracy"],
        precision=rates["precision"],
        recall=rates["recall"],
        f1=rates["f1"],
        far=rates["far"],
        frr=rates["frr"],
        auc=auc,
        eer=eer,
        eer_threshold=eer_threshold,
        confusion=confusion,
        roc_fpr=fpr,
        roc_tpr=tpr,
        roc_thresholds=thresholds,
        num_pairs=int(labels.size),
        num_positive=int(labels.sum()),
        num_negative=int((~labels).sum()),
    )


class RecognitionMetricCalculator:
    """Facade for recognition / verification metric computation."""

    def compute(
        self,
        y_true: np.ndarray | list[int] | list[bool],
        y_score: np.ndarray | list[float],
        *,
        threshold: float | None = None,
    ) -> RecognitionMetrics:
        """Compute recognition metrics.

        Args:
            y_true: Ground-truth same/different labels.
            y_score: Similarity scores.
            threshold: Optional fixed threshold (defaults to EER threshold).

        Returns:
            :class:`RecognitionMetrics`.
        """
        return compute_recognition_metrics(
            np.asarray(y_true),
            np.asarray(y_score, dtype=np.float64),
            threshold=threshold,
        )
