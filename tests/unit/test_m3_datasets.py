"""Unit tests for Milestone M3 dataset adapters."""

from __future__ import annotations

from pathlib import Path

from facebench.datasets import (
    AgeDBDataset,
    ARFaceDataset,
    CFPFPDataset,
    ChokePointDataset,
    CPLFWDataset,
    DatasetFactory,
    TinyFaceDataset,
    YTFDataset,
)
from tests.fixtures.m3_synthetic import (
    make_synthetic_agedb_flat,
    make_synthetic_ar_face,
    make_synthetic_cfp_fp,
    make_synthetic_chokepoint,
    make_synthetic_cplfw,
    make_synthetic_tinyface,
    make_synthetic_ytf,
)


def test_cfp_fp_adapter(tmp_path: Path) -> None:
    """CFP-FP loads recursive pose images and path/label pairs."""
    root = make_synthetic_cfp_fp(tmp_path / "cfp")
    ds = CFPFPDataset(root)
    assert ds.validate_integrity().ok
    index = ds.load_dataset()
    assert index.image_count == 8
    pairs = ds.load_identity_pairs()
    assert len(pairs) == 2
    assert pairs[0].issame is True
    assert pairs[0].sample_a.metadata.get("pose_split") == "frontal"


def test_cplfw_adapter(tmp_path: Path) -> None:
    """CPLFW parses LFW-style pairs."""
    root = make_synthetic_cplfw(tmp_path / "cplfw")
    ds = CPLFWDataset(root)
    assert ds.validate_integrity().ok
    assert ds.load_dataset().image_count == 4
    pairs = ds.load_identity_pairs()
    assert pairs[0].issame is True
    assert pairs[1].issame is False


def test_agedb_flat_adapter(tmp_path: Path) -> None:
    """AgeDB flat filenames parse identity/age metadata."""
    root = make_synthetic_agedb_flat(tmp_path / "agedb")
    ds = AgeDBDataset(root)
    assert ds.validate_integrity().ok
    index = ds.load_dataset()
    assert "Maria" in index.identities
    assert index.samples_by_identity["Maria"][0].metadata.get("age") == "35"
    pairs = ds.load_identity_pairs()
    assert len(pairs) == 2


def test_tinyface_gallery_probe(tmp_path: Path) -> None:
    """TinyFace loads Gallery/Probe splits and synthesizes pairs."""
    root = make_synthetic_tinyface(tmp_path / "tinyface")
    ds = TinyFaceDataset(root)
    assert ds.validate_integrity().ok
    gallery = ds.load_gallery()
    probe = ds.load_probe()
    assert len(gallery) == 2
    assert len(probe) == 2
    assert all(s.metadata.get("split") == "gallery" for s in gallery)
    pairs = ds.load_identity_pairs()
    assert any(p.issame for p in pairs)
    assert any(not p.issame for p in pairs)


def test_ar_face_synthesized_pairs(tmp_path: Path) -> None:
    """AR Face indexes occlusion cues and synthesizes pairs."""
    root = make_synthetic_ar_face(tmp_path / "ar")
    ds = ARFaceDataset(root)
    assert ds.validate_integrity().ok
    index = ds.load_dataset()
    cues = [
        s.metadata.get("occlusion_cue")
        for samples in index.samples_by_identity.values()
        for s in samples
    ]
    assert True in cues
    pairs = ds.load_identity_pairs()
    assert len(pairs) >= 2


def test_chokepoint_recursive_index(tmp_path: Path) -> None:
    """ChokePoint indexes nested camera folders."""
    root = make_synthetic_chokepoint(tmp_path / "choke")
    ds = ChokePointDataset(root)
    assert ds.validate_integrity().ok
    index = ds.load_dataset()
    assert index.image_count == 4
    pairs = ds.load_identity_pairs()
    assert len(pairs) >= 2


def test_ytf_video_pair_resolution(tmp_path: Path) -> None:
    """YTF resolves video directories in pairs to representative frames."""
    root = make_synthetic_ytf(tmp_path / "ytf")
    ds = YTFDataset(root)
    assert ds.validate_integrity().ok
    index = ds.load_dataset()
    assert index.image_count == 6
    pairs = ds.load_identity_pairs()
    assert len(pairs) == 2
    assert pairs[0].sample_a.path.is_file()
    assert pairs[0].sample_a.metadata.get("video_dir")


def test_factory_creates_all_eight(tmp_path: Path) -> None:
    """DatasetFactory constructs every v1 public adapter."""
    factory = DatasetFactory()
    assert factory.available() == sorted(
        [
            "AgeDB-30",
            "AR-Face",
            "CFP-FP",
            "CPLFW",
            "ChokePoint",
            "LFW",
            "TinyFace",
            "YTF",
        ]
    )
    mapping = {
        "CFP-FP": make_synthetic_cfp_fp(tmp_path / "cfp"),
        "CPLFW": make_synthetic_cplfw(tmp_path / "cplfw"),
        "AgeDB-30": make_synthetic_agedb_flat(tmp_path / "agedb"),
        "TinyFace": make_synthetic_tinyface(tmp_path / "tiny"),
        "AR-Face": make_synthetic_ar_face(tmp_path / "ar"),
        "ChokePoint": make_synthetic_chokepoint(tmp_path / "choke"),
        "YTF": make_synthetic_ytf(tmp_path / "ytf"),
    }
    for name, root in mapping.items():
        ds = factory.create(name, root)
        assert ds.validate_integrity().ok
        assert ds.load_dataset().image_count > 0


def test_prep_docs_exist() -> None:
    """Each M3 dataset has a preparation guide on disk."""
    root = Path(__file__).resolve().parents[2]
    for name in (
        "cfp_fp.md",
        "cplfw.md",
        "agedb.md",
        "tinyface.md",
        "ar_face.md",
        "chokepoint.md",
        "ytf.md",
    ):
        assert (root / "docs" / "datasets" / name).is_file()
