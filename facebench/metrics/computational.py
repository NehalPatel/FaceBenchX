"""Computational / resource metric collection."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil


@dataclass(slots=True)
class ComputationalMetrics:
    """Resource and latency statistics for a recognition workload.

    Attributes:
        model_load_time_s: Model loading wall time in seconds.
        avg_inference_time_s: Mean per-image inference time.
        avg_embedding_time_s: Mean embedding generation time.
        recognition_latency_s: End-to-end mean latency per comparison/sample.
        throughput_fps: Samples processed per second.
        cpu_percent: Mean process CPU utilization percentage.
        ram_rss_mb: Peak / final resident set size in MiB.
        gpu_percent: Mean GPU utilization when available.
        gpu_memory_mb: Peak GPU memory used in MiB when available.
        model_size_mb: On-disk model size in MiB when provided.
        num_samples: Number of timed samples.
        warmup_samples: Warm-up iterations excluded from averages.
        extra: Extensible metadata.
    """

    model_load_time_s: float | None = None
    avg_inference_time_s: float | None = None
    avg_embedding_time_s: float | None = None
    recognition_latency_s: float | None = None
    throughput_fps: float | None = None
    cpu_percent: float | None = None
    ram_rss_mb: float | None = None
    gpu_percent: float | None = None
    gpu_memory_mb: float | None = None
    model_size_mb: float | None = None
    num_samples: int = 0
    warmup_samples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            JSON-friendly metrics mapping.
        """
        return asdict(self)


def model_size_mb(path: str | Path | None) -> float | None:
    """Compute on-disk model size in MiB.

    Args:
        path: File or directory containing model weights.

    Returns:
        Size in MiB, or ``None`` if ``path`` is missing.
    """
    if path is None:
        return None
    target = Path(path).expanduser()
    if not target.exists():
        return None
    if target.is_file():
        return target.stat().st_size / (1024 * 1024)
    total = 0
    for file_path in target.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total / (1024 * 1024)


def _gpu_stats() -> tuple[float | None, float | None]:
    """Best-effort GPU utilization and memory readout.

    Returns:
        ``(gpu_percent, gpu_memory_mb)`` or ``(None, None)``.
    """
    try:
        import pynvml
    except ImportError:
        return None, None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_percent = float(util.gpu)
        gpu_memory_mb = float(memory.used) / (1024 * 1024)
        pynvml.nvmlShutdown()
        return gpu_percent, gpu_memory_mb
    except Exception:  # noqa: BLE001 - optional probe
        return None, None


class ComputeProfiler:
    """Profile latency and resource usage around recognition workloads.

    Warm-up iterations are recorded but excluded from averages to reduce
    cold-start bias (see design NFR-07).
    """

    def __init__(self, *, warmup: int = 2) -> None:
        """Initialize the profiler.

        Args:
            warmup: Number of leading samples excluded from averages.
        """
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        self.warmup = warmup
        self._inference_times: list[float] = []
        self._embedding_times: list[float] = []
        self._latencies: list[float] = []
        self._cpu_samples: list[float] = []
        self._rss_samples: list[float] = []
        self._gpu_util_samples: list[float] = []
        self._gpu_mem_samples: list[float] = []
        self._model_load_time_s: float | None = None
        self._model_size_mb: float | None = None
        self._process = psutil.Process(os.getpid())
        # Prime CPU percent baseline.
        self._process.cpu_percent(interval=None)

    def record_model_load(self, seconds: float) -> None:
        """Record model loading time.

        Args:
            seconds: Wall-clock seconds spent loading.
        """
        self._model_load_time_s = float(seconds)

    def set_model_size_mb(self, size_mb: float | None) -> None:
        """Set on-disk model size.

        Args:
            size_mb: Size in MiB.
        """
        self._model_size_mb = size_mb

    def time_model_load(self, fn: Callable[[], Any]) -> Any:
        """Time a model-loading callable.

        Args:
            fn: Zero-arg callable that loads a model.

        Returns:
            The callable's return value.
        """
        started = time.perf_counter()
        result = fn()
        self.record_model_load(time.perf_counter() - started)
        return result

    def record_embedding_time(self, seconds: float) -> None:
        """Record a single embedding generation duration."""
        self._embedding_times.append(float(seconds))
        self._sample_resources()

    def record_inference_time(self, seconds: float) -> None:
        """Record a single inference duration."""
        self._inference_times.append(float(seconds))
        self._sample_resources()

    def record_latency(self, seconds: float) -> None:
        """Record end-to-end recognition latency for one unit of work."""
        self._latencies.append(float(seconds))
        self._sample_resources()

    def track_embedding(self, fn: Callable[[], Any]) -> Any:
        """Time an embedding callable and record the duration.

        Args:
            fn: Zero-arg embedding function.

        Returns:
            The callable's return value.
        """
        started = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - started
        self.record_embedding_time(elapsed)
        self.record_inference_time(elapsed)
        self.record_latency(elapsed)
        return result

    def summarize(self) -> ComputationalMetrics:
        """Aggregate recorded timings into :class:`ComputationalMetrics`.

        Returns:
            Summarized computational metrics.
        """
        emb = self._mean_after_warmup(self._embedding_times)
        inf = self._mean_after_warmup(self._inference_times)
        lat = self._mean_after_warmup(self._latencies)
        timed = self._values_after_warmup(self._latencies or self._embedding_times)
        throughput = (1.0 / lat) if lat and lat > 0 else None
        return ComputationalMetrics(
            model_load_time_s=self._model_load_time_s,
            avg_inference_time_s=inf,
            avg_embedding_time_s=emb,
            recognition_latency_s=lat,
            throughput_fps=throughput,
            cpu_percent=self._mean_or_none(self._cpu_samples),
            ram_rss_mb=max(self._rss_samples) if self._rss_samples else None,
            gpu_percent=self._mean_or_none(self._gpu_util_samples),
            gpu_memory_mb=(
                max(self._gpu_mem_samples) if self._gpu_mem_samples else None
            ),
            model_size_mb=self._model_size_mb,
            num_samples=len(timed),
            warmup_samples=self.warmup,
        )

    def _sample_resources(self) -> None:
        """Sample CPU / RAM / GPU counters."""
        self._cpu_samples.append(float(self._process.cpu_percent(interval=None)))
        self._rss_samples.append(self._process.memory_info().rss / (1024 * 1024))
        gpu_percent, gpu_memory = _gpu_stats()
        if gpu_percent is not None:
            self._gpu_util_samples.append(gpu_percent)
        if gpu_memory is not None:
            self._gpu_mem_samples.append(gpu_memory)

    def _values_after_warmup(self, values: list[float]) -> list[float]:
        if len(values) <= self.warmup:
            return list(values)
        return values[self.warmup :]

    def _mean_after_warmup(self, values: list[float]) -> float | None:
        subset = self._values_after_warmup(values)
        if not subset:
            return None
        return float(sum(subset) / len(subset))

    @staticmethod
    def _mean_or_none(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))
