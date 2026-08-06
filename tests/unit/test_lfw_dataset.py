"""Unit tests for the LFW dataset adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from facebench.datasets.lfw import LFWDataset
from tests.fixtures.lfw_synthetic import make_synthetic_lfw


def test_load_dataset_indexes_identities(tmp_path: Path) -> None:
    """load_dataset discovers identities and images."""
    root = make_synthetic_lfw(tmp_path / "lfw")
    ds = LFWDataset(root)
    index = ds.load_dataset()
    assert index.image_count == 7
    assert index.identities == ["Alice_Example", "Bob_Example", "Carol_Example"]
    assert len(index.samples_by_identity["Alice_Example"]) == 3


def test_load_identity_pairs(tmp_path: Path) -> None:
    """load_identity_pairs parses same/different protocol lines."""
    root = make_synthetic_lfw(tmp_path / "lfw")
    ds = LFWDataset(root)
    pairs = ds.load_identity_pairs()
    assert len(pairs) == 4
    assert pairs[0].issame is True
    assert pairs[0].sample_a.identity == "Alice_Example"
    assert pairs[2].issame is False
    assert pairs[0].sample_a.path.name == "Alice_Example_0001.jpg"
    assert pairs[0].sample_b.path.name == "Alice_Example_0002.jpg"


def test_pairs_beside_root(tmp_path: Path) -> None:
    """pairs.txt beside the identity tree is discovered."""
    root = make_synthetic_lfw(tmp_path / "lfw", pairs_beside_root=True)
    ds = LFWDataset(root)
    assert ds.resolve_pairs_file() is not None
    assert ds.validate_integrity().ok is True
    assert len(ds.load_identity_pairs()) == 4


def test_validate_integrity_success_and_failure(tmp_path: Path) -> None:
    """validate_integrity passes on complete trees and fails when pairs missing."""
    good_root = make_synthetic_lfw(tmp_path / "good")
    good = LFWDataset(good_root).validate_integrity()
    assert good.ok is True
    assert good.prep_doc is None

    bad_root = make_synthetic_lfw(tmp_path / "bad", include_pairs=False)
    bad = LFWDataset(bad_root).validate_integrity()
    assert bad.ok is False
    assert bad.prep_doc == "docs/datasets/lfw.md"


def test_preprocess_resolves_relative_paths(tmp_path: Path) -> None:
    """preprocess resolves relative paths against the dataset root."""
    root = make_synthetic_lfw(tmp_path / "lfw")
    ds = LFWDataset(root)
    from facebench.datasets.types import Sample

    raw = Sample(
        path=Path("Alice_Example/Alice_Example_0001.jpg"),
        identity=" Alice_Example ",
    )
    out = ds.preprocess(raw)
    assert out.path.is_absolute()
    assert out.path.is_file()
    assert out.identity == "Alice_Example"
    assert out.metadata["exists"] is True


def test_gallery_and_probe_split(tmp_path: Path) -> None:
    """Gallery takes first image; probe takes the remainder."""
    root = make_synthetic_lfw(tmp_path / "lfw")
    ds = LFWDataset(root)
    gallery = ds.load_gallery()
    probe = ds.load_probe()
    assert len(gallery) == 3
    assert len(probe) == 4  # 7 total - 3 gallery
    gallery_ids = {(s.identity, s.path.name) for s in gallery}
    probe_ids = {(s.identity, s.path.name) for s in probe}
    assert gallery_ids.isdisjoint(probe_ids)


def test_missing_pairs_file_raises(tmp_path: Path) -> None:
    """load_identity_pairs raises when no pairs file exists."""
    root = make_synthetic_lfw(tmp_path / "lfw", include_pairs=False)
    ds = LFWDataset(root)
    with pytest.raises(FileNotFoundError, match="pairs"):
        ds.load_identity_pairs()
