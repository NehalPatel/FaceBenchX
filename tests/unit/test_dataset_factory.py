"""Unit tests for DatasetFactory."""

from __future__ import annotations

from pathlib import Path

import pytest

from facebench.datasets.factory import DatasetFactory, DatasetFactoryError
from facebench.datasets.lfw import LFWDataset
from tests.fixtures.lfw_synthetic import make_synthetic_lfw
from tests.fixtures.m3_synthetic import make_synthetic_agedb_flat


def test_factory_creates_lfw(tmp_path: Path) -> None:
    """LFW alias and canonical name construct LFWDataset."""
    root = make_synthetic_lfw(tmp_path / "lfw")
    factory = DatasetFactory()
    ds = factory.create("lfw", root)
    assert isinstance(ds, LFWDataset)
    assert ds.name == "LFW"
    assert "LFW" in factory.available()


def test_factory_creates_agedb(tmp_path: Path) -> None:
    """AgeDB-30 is constructible after Milestone M3."""
    root = make_synthetic_agedb_flat(tmp_path / "agedb")
    ds = DatasetFactory().create("AgeDB-30", root)
    assert ds.name == "AgeDB-30"
    assert ds.validate_integrity().ok


def test_factory_rejects_unknown_dataset(tmp_path: Path) -> None:
    """Custom datasets are not constructible."""
    factory = DatasetFactory()
    with pytest.raises(DatasetFactoryError, match="Unknown dataset"):
        factory.create("MyPrivateSet", tmp_path)


def test_factory_lists_all_eight() -> None:
    """All eight v1 public adapters are registered."""
    assert len(DatasetFactory().available()) == 8
