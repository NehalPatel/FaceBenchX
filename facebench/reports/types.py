"""Report payload types for FaceBench experiment outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from facebench.metrics.computational import ComputationalMetrics
from facebench.metrics.recognition import RecognitionMetrics


@dataclass(slots=True)
class ExperimentReportData:
    """Structured inputs for per-experiment report generation.

    Attributes:
        experiment_id: Unique experiment identifier.
        experiment_name: Human-readable name.
        dataset_name: Evaluated dataset.
        model_name: Evaluated model.
        config: Snapshot of experiment configuration.
        recognition: Recognition metrics (optional).
        computational: Computational metrics (optional).
        figure_paths: Relative/absolute paths to generated figures.
        notes: Free-form notes.
        extra: Extensible metadata.
    """

    experiment_id: str
    experiment_name: str
    dataset_name: str
    model_name: str
    config: dict[str, Any] = field(default_factory=dict)
    recognition: RecognitionMetrics | None = None
    computational: ComputationalMetrics | None = None
    figure_paths: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        payload: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "config": self.config,
            "figure_paths": self.figure_paths,
            "notes": self.notes,
            "extra": self.extra,
            "recognition": self.recognition.to_dict() if self.recognition else None,
            "computational": (
                self.computational.to_dict() if self.computational else None
            ),
        }
        return payload


@dataclass(slots=True)
class ComparisonRow:
    """One row in an aggregated cross-model / cross-dataset comparison."""

    dataset_name: str
    model_name: str
    accuracy: float | None = None
    f1: float | None = None
    auc: float | None = None
    eer: float | None = None
    avg_embedding_time_s: float | None = None
    throughput_fps: float | None = None
    cpu_percent: float | None = None
    ram_rss_mb: float | None = None
    gpu_percent: float | None = None
    gpu_memory_mb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the row."""
        return asdict(self)


def comparison_row_from_report(data: ExperimentReportData) -> ComparisonRow:
    """Build a comparison row from a single experiment report payload.

    Args:
        data: Per-experiment report data.

    Returns:
        :class:`ComparisonRow`.
    """
    recog = data.recognition
    comp = data.computational
    return ComparisonRow(
        dataset_name=data.dataset_name,
        model_name=data.model_name,
        accuracy=None if recog is None else recog.accuracy,
        f1=None if recog is None else recog.f1,
        auc=None if recog is None else recog.auc,
        eer=None if recog is None else recog.eer,
        avg_embedding_time_s=None if comp is None else comp.avg_embedding_time_s,
        throughput_fps=None if comp is None else comp.throughput_fps,
        cpu_percent=None if comp is None else comp.cpu_percent,
        ram_rss_mb=None if comp is None else comp.ram_rss_mb,
        gpu_percent=None if comp is None else comp.gpu_percent,
        gpu_memory_mb=None if comp is None else comp.gpu_memory_mb,
    )
