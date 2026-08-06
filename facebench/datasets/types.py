"""Shared dataset data structures for FaceBench adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Sample:
    """A single image sample in a face recognition dataset.

    Attributes:
        path: Absolute or normalized filesystem path to the image.
        identity: Identity / subject label.
        image_id: Optional unique image identifier within the dataset.
        metadata: Extensible key/value metadata.
    """

    path: Path
    identity: str
    image_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IdentityPair:
    """A verification pair with a same/different ground-truth label.

    Attributes:
        sample_a: First sample in the pair.
        sample_b: Second sample in the pair.
        issame: ``True`` if both samples share an identity.
        fold: Optional fold index for cross-validation protocols.
        metadata: Extensible key/value metadata.
    """

    sample_a: Sample
    sample_b: Sample
    issame: bool
    fold: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetIndex:
    """Indexed view of identities and images discovered on disk.

    Attributes:
        root_path: Dataset root used for discovery.
        identities: Sorted identity labels.
        samples_by_identity: Identity → list of samples.
        image_count: Total number of indexed images.
        metadata: Extensible index-level metadata.
    """

    root_path: Path
    identities: list[str]
    samples_by_identity: dict[str, list[Sample]]
    image_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    """Result of a dataset integrity / layout validation check.

    Attributes:
        ok: Whether the dataset is usable.
        missing: Paths or requirements that were not found.
        messages: Human-readable diagnostic messages.
        prep_doc: Relative documentation path for preparation guidance.
        checked_path: Root path that was validated.
    """

    ok: bool
    missing: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    prep_doc: str | None = None
    checked_path: str | None = None
