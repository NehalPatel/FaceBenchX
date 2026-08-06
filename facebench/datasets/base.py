"""Abstract dataset adapter contract for FaceBench public benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from facebench.datasets.types import (
    DatasetIndex,
    IdentityPair,
    Sample,
    ValidationResult,
)


class BaseDataset(ABC):
    """Common interface implemented by every public dataset adapter.

    Adapters are responsible for local path discovery and protocol loading
    only. Face detection, recognition models, and metrics live elsewhere.
    """

    #: Canonical dataset name (e.g. ``"LFW"``).
    name: str = "BASE"
    #: Research category label (e.g. ``"general"``).
    category: str = "unknown"
    #: Documentation path describing how to prepare the dataset.
    prep_doc: str = "docs/datasets/README.md"

    def __init__(self, root_path: str | Path) -> None:
        """Initialize the adapter with a local dataset root.

        Args:
            root_path: Filesystem path supplied via YAML configuration.
                Datasets are never downloaded or bundled by FaceBench.
        """
        self.root_path = Path(root_path).expanduser().resolve()
        self._index: DatasetIndex | None = None

    @abstractmethod
    def load_dataset(self) -> DatasetIndex:
        """Discover and index images / identities under ``root_path``.

        Returns:
            Populated :class:`DatasetIndex`.
        """

    @abstractmethod
    def load_identity_pairs(self) -> list[IdentityPair]:
        """Load verification pairs (same / different identities).

        Returns:
            List of :class:`IdentityPair` entries.
        """

    @abstractmethod
    def load_gallery(self) -> list[Sample]:
        """Load the enrollment / gallery set.

        Returns:
            Gallery samples.
        """

    @abstractmethod
    def load_probe(self) -> list[Sample]:
        """Load the probe / query set.

        Returns:
            Probe samples.
        """

    @abstractmethod
    def preprocess(self, sample: Sample) -> Sample:
        """Apply dataset-specific path / label normalization.

        Args:
            sample: Raw sample from an index or pair list.

        Returns:
            Normalized sample ready for shared detect/align stages.
        """

    @abstractmethod
    def validate_integrity(self) -> ValidationResult:
        """Validate that the local dataset layout is usable.

        Returns:
            :class:`ValidationResult` including missing-file hints and
            a pointer to the preparation guide when validation fails.
        """

    def get_index(self, *, reload: bool = False) -> DatasetIndex:
        """Return a cached dataset index, loading it on first use.

        Args:
            reload: When ``True``, force re-discovery from disk.

        Returns:
            Cached or freshly loaded :class:`DatasetIndex`.
        """
        if self._index is None or reload:
            self._index = self.load_dataset()
        return self._index
