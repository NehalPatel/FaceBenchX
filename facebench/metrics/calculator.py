"""Unified metric calculator facade."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from facebench.metrics.computational import (
    ComputationalMetrics,
    ComputeProfiler,
    model_size_mb,
)
from facebench.metrics.recognition import (
    RecognitionMetricCalculator,
    RecognitionMetrics,
)


class MetricCalculator:
    """Facade combining recognition and computational metric helpers."""

    def __init__(self, *, warmup: int = 2) -> None:
        """Initialize calculators.

        Args:
            warmup: Default warm-up samples for a new profiler.
        """
        self._recognition = RecognitionMetricCalculator()
        self._warmup = warmup

    def recognition(
        self,
        y_true: np.ndarray | list[int] | list[bool],
        y_score: np.ndarray | list[float],
        *,
        threshold: float | None = None,
    ) -> RecognitionMetrics:
        """Compute recognition / verification metrics.

        Args:
            y_true: Same/different labels.
            y_score: Similarity scores.
            threshold: Optional fixed threshold.

        Returns:
            :class:`RecognitionMetrics`.
        """
        return self._recognition.compute(y_true, y_score, threshold=threshold)

    def computational(self, profiler: ComputeProfiler) -> ComputationalMetrics:
        """Summarize a :class:`ComputeProfiler` recording.

        Args:
            profiler: Profiler with recorded timings.

        Returns:
            :class:`ComputationalMetrics`.
        """
        return profiler.summarize()

    def create_profiler(self, *, warmup: int | None = None) -> ComputeProfiler:
        """Create a compute profiler.

        Args:
            warmup: Optional warm-up override.

        Returns:
            New :class:`ComputeProfiler`.
        """
        return ComputeProfiler(warmup=self._warmup if warmup is None else warmup)

    @staticmethod
    def model_size(path: str | Path | None) -> float | None:
        """Return on-disk model size in MiB."""
        return model_size_mb(path)
