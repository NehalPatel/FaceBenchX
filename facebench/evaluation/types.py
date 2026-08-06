"""Shared result types for benchmark execution protocols."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from facebench.metrics.computational import ComputationalMetrics
from facebench.metrics.recognition import RecognitionMetrics


@dataclass(slots=True)
class VerificationResult:
    """Outcome of a verification (1:1) protocol run."""

    recognition: RecognitionMetrics
    computational: ComputationalMetrics
    labels: list[int]
    scores: list[float]
    num_pairs: int
    transform: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON reports."""
        return {
            "recognition": self.recognition.to_dict(),
            "computational": self.computational.to_dict(),
            "num_pairs": self.num_pairs,
            "transform": self.transform,
            "extra": self.extra,
        }


@dataclass(slots=True)
class IdentificationResult:
    """Outcome of an identification (1:N) protocol run."""

    rank1_accuracy: float
    cmc: dict[int, float]
    mean_search_latency_s: float
    throughput_probes_per_s: float
    num_probes: int
    num_gallery: int
    num_identities: int
    skipped_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON reports."""
        payload = asdict(self)
        return payload


@dataclass(slots=True)
class RobustnessConditionResult:
    """Verification metrics under one synthetic degradation."""

    transform: str
    recognition: RecognitionMetrics
    delta_accuracy: float | None = None
    delta_auc: float | None = None
    delta_eer: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON reports."""
        return {
            "transform": self.transform,
            "recognition": self.recognition.to_dict(),
            "delta_accuracy": self.delta_accuracy,
            "delta_auc": self.delta_auc,
            "delta_eer": self.delta_eer,
            "extra": self.extra,
        }


@dataclass(slots=True)
class RobustnessReport:
    """Baseline plus per-transform robustness results (synthetic, public base)."""

    baseline: VerificationResult
    conditions: list[RobustnessConditionResult]
    base_dataset: str
    notes: str = (
        "Synthetic robustness on a public base dataset; "
        "primary RQ3 evidence remains category public sets."
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON reports."""
        return {
            "base_dataset": self.base_dataset,
            "notes": self.notes,
            "baseline": self.baseline.to_dict(),
            "conditions": [item.to_dict() for item in self.conditions],
        }


@dataclass(slots=True)
class ScalabilityPoint:
    """One gallery-size operating point on the scalability ladder."""

    identity_count: int
    identification: IdentificationResult | None
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON reports."""
        return {
            "identity_count": self.identity_count,
            "skipped_reason": self.skipped_reason,
            "identification": (
                None if self.identification is None else self.identification.to_dict()
            ),
        }


@dataclass(slots=True)
class ScalabilityReport:
    """Scalability ladder results across enrolled identity counts."""

    points: list[ScalabilityPoint]
    dataset_name: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON reports."""
        return {
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "points": [point.to_dict() for point in self.points],
        }
