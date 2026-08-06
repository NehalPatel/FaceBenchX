"""Unit tests for computational metrics and matchers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from facebench.matcher import CosineMatcher, EuclideanMatcher, create_matcher
from facebench.metrics import ComputeProfiler, MetricCalculator, model_size_mb


def test_compute_profiler_warmup_and_throughput() -> None:
    """Profiler excludes warm-up samples and reports positive FPS."""
    profiler = ComputeProfiler(warmup=2)
    profiler.record_model_load(0.05)
    for _ in range(5):
        profiler.track_embedding(lambda: time.sleep(0.001) or 1)
    metrics = profiler.summarize()
    assert metrics.model_load_time_s == pytest.approx(0.05)
    assert metrics.num_samples == 3
    assert metrics.warmup_samples == 2
    assert metrics.avg_embedding_time_s is not None
    assert metrics.throughput_fps is not None
    assert metrics.throughput_fps > 0
    assert metrics.ram_rss_mb is not None


def test_model_size_mb(tmp_path: Path) -> None:
    """model_size_mb reports file size in MiB."""
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"x" * (2 * 1024 * 1024))
    size = model_size_mb(weights)
    assert size == pytest.approx(2.0, abs=0.01)
    assert model_size_mb(tmp_path / "missing") is None


def test_metric_calculator_computational(tmp_path: Path) -> None:
    """Facade summarizes profiler output and model size."""
    calc = MetricCalculator(warmup=0)
    profiler = calc.create_profiler()
    profiler.set_model_size_mb(calc.model_size(tmp_path))
    profiler.record_latency(0.01)
    profiler.record_latency(0.03)
    metrics = calc.computational(profiler)
    assert metrics.recognition_latency_s == pytest.approx(0.02)
    assert "avg_inference_time_s" in metrics.to_dict()


def test_matchers_cosine_and_euclidean() -> None:
    """Matchers score identical vectors highest."""
    a = np.asarray([1.0, 0.0, 0.0])
    b = np.asarray([1.0, 0.0, 0.0])
    c = np.asarray([0.0, 1.0, 0.0])
    cosine = create_matcher("cosine")
    euclid = create_matcher("euclidean")
    assert isinstance(cosine, CosineMatcher)
    assert isinstance(euclid, EuclideanMatcher)
    assert cosine.score(a, b) == pytest.approx(1.0)
    assert cosine.decide(a, b, threshold=0.5) is True
    assert euclid.score(a, b) > euclid.score(a, c)
