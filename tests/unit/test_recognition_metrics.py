"""Unit tests for recognition metrics."""

from __future__ import annotations

import numpy as np
import pytest

from facebench.metrics import (
    MetricCalculator,
    compute_recognition_metrics,
    confusion_at_threshold,
    equal_error_rate,
    roc_auc,
    roc_curve,
)


def test_perfect_separation_metrics() -> None:
    """Perfectly separable scores yield AUC=1 and zero errors at mid threshold."""
    y_true = np.asarray([1, 1, 1, 0, 0, 0])
    y_score = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    metrics = compute_recognition_metrics(y_true, y_score, threshold=0.5)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.far == pytest.approx(0.0)
    assert metrics.frr == pytest.approx(0.0)
    assert metrics.auc == pytest.approx(1.0)
    assert metrics.confusion.as_matrix().tolist() == [[3, 0], [0, 3]]


def test_confusion_and_eer() -> None:
    """Confusion counts and EER are finite for overlapping scores."""
    y_true = [1, 1, 0, 0, 1, 0]
    y_score = [0.6, 0.4, 0.55, 0.2, 0.7, 0.65]
    confusion = confusion_at_threshold(y_true, y_score, threshold=0.5)
    assert confusion.true_positive + confusion.false_negative == 3
    eer, thr = equal_error_rate(y_true, y_score)
    assert 0.0 <= eer <= 1.0
    assert isinstance(thr, float)


def test_roc_curve_shapes() -> None:
    """ROC arrays are aligned and AUC is in range."""
    y_true = np.asarray([1, 0, 1, 0, 1, 0])
    y_score = np.asarray([0.9, 0.8, 0.4, 0.3, 0.7, 0.1])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    assert fpr.shape == tpr.shape
    assert thresholds.size == fpr.size
    auc = roc_auc(fpr, tpr)
    assert 0.0 <= auc <= 1.0


def test_metric_calculator_recognition() -> None:
    """Facade returns serializable recognition metrics."""
    calc = MetricCalculator()
    metrics = calc.recognition([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], threshold=0.5)
    payload = metrics.to_dict()
    assert payload["accuracy"] == pytest.approx(1.0)
    assert "confusion_matrix" in payload
    assert isinstance(payload["roc_fpr"], list)
