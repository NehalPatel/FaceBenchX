"""Contract checks for all v1 public dataset adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from facebench.datasets.base import BaseDataset
from facebench.datasets.factory import DatasetFactory
from tests.fixtures.lfw_synthetic import make_synthetic_lfw
from tests.fixtures.m3_synthetic import (
    make_synthetic_agedb_flat,
    make_synthetic_ar_face,
    make_synthetic_cfp_fp,
    make_synthetic_chokepoint,
    make_synthetic_cplfw,
    make_synthetic_tinyface,
    make_synthetic_ytf,
)

REQUIRED_METHODS = (
    "load_dataset",
    "load_identity_pairs",
    "load_gallery",
    "load_probe",
    "preprocess",
    "validate_integrity",
)


@pytest.fixture
def all_dataset_roots(tmp_path: Path) -> dict[str, Path]:
    """Build synthetic roots for every public adapter."""
    return {
        "LFW": make_synthetic_lfw(tmp_path / "lfw"),
        "CFP-FP": make_synthetic_cfp_fp(tmp_path / "cfp"),
        "CPLFW": make_synthetic_cplfw(tmp_path / "cplfw"),
        "AgeDB-30": make_synthetic_agedb_flat(tmp_path / "agedb"),
        "TinyFace": make_synthetic_tinyface(tmp_path / "tiny"),
        "AR-Face": make_synthetic_ar_face(tmp_path / "ar"),
        "ChokePoint": make_synthetic_chokepoint(tmp_path / "choke"),
        "YTF": make_synthetic_ytf(tmp_path / "ytf"),
    }


@pytest.mark.parametrize(
    "name",
    ["LFW", "CFP-FP", "CPLFW", "AgeDB-30", "TinyFace", "AR-Face", "ChokePoint", "YTF"],
)
def test_all_adapters_satisfy_contract(
    name: str, all_dataset_roots: dict[str, Path]
) -> None:
    """Every public adapter implements BaseDataset and basic load APIs."""
    ds = DatasetFactory().create(name, all_dataset_roots[name])
    assert isinstance(ds, BaseDataset)
    for method_name in REQUIRED_METHODS:
        assert callable(getattr(ds, method_name))
    assert ds.validate_integrity().ok is True
    assert ds.load_dataset().image_count > 0
    assert len(ds.load_identity_pairs()) > 0
    assert len(ds.load_gallery()) > 0
