"""Recognition and computational metrics (Milestone / Phase 6)."""

from __future__ import annotations

from facebench.metrics.calculator import MetricCalculator
from facebench.metrics.computational import (
    ComputationalMetrics,
    ComputeProfiler,
    model_size_mb,
)
from facebench.metrics.recognition import (
    ConfusionCounts,
    RecognitionMetricCalculator,
    RecognitionMetrics,
    binary_rates,
    compute_recognition_metrics,
    confusion_at_threshold,
    equal_error_rate,
    roc_auc,
    roc_curve,
)

__all__ = [
    "ComputationalMetrics",
    "ComputeProfiler",
    "ConfusionCounts",
    "MetricCalculator",
    "RecognitionMetricCalculator",
    "RecognitionMetrics",
    "binary_rates",
    "compute_recognition_metrics",
    "confusion_at_threshold",
    "equal_error_rate",
    "model_size_mb",
    "roc_auc",
    "roc_curve",
]
