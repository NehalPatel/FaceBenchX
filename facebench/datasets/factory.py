"""Factory for constructing public dataset adapters from config names."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from facebench.core.registry import CategoryRegistry, get_default_registry
from facebench.datasets.agedb.dataset import AgeDBDataset
from facebench.datasets.ar_face.dataset import ARFaceDataset
from facebench.datasets.base import BaseDataset
from facebench.datasets.cfp_fp.dataset import CFPFPDataset
from facebench.datasets.chokepoint.dataset import ChokePointDataset
from facebench.datasets.cplfw.dataset import CPLFWDataset
from facebench.datasets.lfw.dataset import LFWDataset
from facebench.datasets.tinyface.dataset import TinyFaceDataset
from facebench.datasets.ytf.dataset import YTFDataset


class DatasetFactoryError(ValueError):
    """Raised when a dataset adapter cannot be constructed."""


class DatasetFactory:
    """Create :class:`BaseDataset` adapters by canonical dataset name.

    Milestone M3 registers all eight v1 public dataset adapters.
    """

    def __init__(self, registry: CategoryRegistry | None = None) -> None:
        """Initialize the factory.

        Args:
            registry: Category registry used to canonicalize names.
        """
        self._registry = registry or get_default_registry()
        self._builders: dict[str, Callable[..., BaseDataset]] = {
            "LFW": self._build_lfw,
            "CFP-FP": self._build_cfp_fp,
            "CPLFW": self._build_cplfw,
            "AgeDB-30": self._build_agedb,
            "TinyFace": self._build_tinyface,
            "AR-Face": self._build_ar_face,
            "ChokePoint": self._build_chokepoint,
            "YTF": self._build_ytf,
        }

    def available(self) -> list[str]:
        """Return dataset names that currently have implemented adapters.

        Returns:
            Sorted list of constructible dataset names.
        """
        return sorted(self._builders.keys())

    def create(
        self,
        name: str,
        root_path: str | Path,
        **kwargs: Any,
    ) -> BaseDataset:
        """Construct a dataset adapter.

        Args:
            name: Canonical dataset name or known alias.
            root_path: Local dataset root path from YAML.
            **kwargs: Adapter-specific options (e.g. ``pairs_file``).

        Returns:
            Concrete :class:`BaseDataset` instance.

        Raises:
            DatasetFactoryError: If the dataset is unknown or not implemented.
        """
        try:
            canonical = self._registry.canonicalize(name)
        except KeyError as exc:
            supported = ", ".join(self._registry.list_all())
            raise DatasetFactoryError(
                f"Unknown dataset {name!r}. Registered public datasets: {supported}"
            ) from exc

        builder = self._builders.get(canonical)
        if builder is None:
            implemented = ", ".join(self.available()) or "(none)"
            raise DatasetFactoryError(
                f"Dataset {canonical!r} is registered but not implemented yet. "
                f"Implemented adapters: {implemented}."
            )
        return builder(root_path=root_path, **kwargs)

    @staticmethod
    def _build_lfw(root_path: str | Path, **kwargs: Any) -> BaseDataset:
        """Build the LFW adapter."""
        return LFWDataset(root_path=root_path, **kwargs)

    @staticmethod
    def _build_cfp_fp(root_path: str | Path, **kwargs: Any) -> BaseDataset:
        """Build the CFP-FP adapter."""
        return CFPFPDataset(root_path=root_path, **kwargs)

    @staticmethod
    def _build_cplfw(root_path: str | Path, **kwargs: Any) -> BaseDataset:
        """Build the CPLFW adapter."""
        return CPLFWDataset(root_path=root_path, **kwargs)

    @staticmethod
    def _build_agedb(root_path: str | Path, **kwargs: Any) -> BaseDataset:
        """Build the AgeDB-30 adapter."""
        return AgeDBDataset(root_path=root_path, **kwargs)

    @staticmethod
    def _build_tinyface(root_path: str | Path, **kwargs: Any) -> BaseDataset:
        """Build the TinyFace adapter."""
        return TinyFaceDataset(root_path=root_path, **kwargs)

    @staticmethod
    def _build_ar_face(root_path: str | Path, **kwargs: Any) -> BaseDataset:
        """Build the AR Face adapter."""
        return ARFaceDataset(root_path=root_path, **kwargs)

    @staticmethod
    def _build_chokepoint(root_path: str | Path, **kwargs: Any) -> BaseDataset:
        """Build the ChokePoint adapter."""
        return ChokePointDataset(root_path=root_path, **kwargs)

    @staticmethod
    def _build_ytf(root_path: str | Path, **kwargs: Any) -> BaseDataset:
        """Build the YTF adapter."""
        return YTFDataset(root_path=root_path, **kwargs)


def create_dataset(
    name: str,
    root_path: str | Path,
    **kwargs: Any,
) -> BaseDataset:
    """Convenience wrapper around :meth:`DatasetFactory.create`.

    Args:
        name: Dataset name or alias.
        root_path: Local dataset root.
        **kwargs: Adapter-specific options.

    Returns:
        Concrete dataset adapter.
    """
    return DatasetFactory().create(name, root_path, **kwargs)
