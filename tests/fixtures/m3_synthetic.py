"""Synthetic fixtures for Milestone M3 public datasets (not real faces)."""

from __future__ import annotations

from pathlib import Path


def _touch_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def make_synthetic_cfp_fp(root: Path) -> Path:
    """Create a minimal CFP-FP-like tree with frontal/profile folders."""
    root.mkdir(parents=True, exist_ok=True)
    for subject in ("001", "002"):
        for pose in ("frontal", "profile"):
            _touch_jpg(root / subject / pose / "01.jpg")
            _touch_jpg(root / subject / pose / "02.jpg")
    pairs = "\n".join(
        [
            "001/frontal/01.jpg 001/profile/01.jpg 1",
            "001/frontal/01.jpg 002/profile/01.jpg 0",
        ]
    )
    (root / "pairs_fp.txt").write_text(pairs + "\n", encoding="utf-8")
    return root


def make_synthetic_cplfw(root: Path) -> Path:
    """Create a minimal CPLFW-like identity tree."""
    root.mkdir(parents=True, exist_ok=True)
    for name, count in (("Alice_X", 2), ("Bob_Y", 2)):
        for idx in range(1, count + 1):
            _touch_jpg(root / name / f"{name}_{idx:04d}.jpg")
    pairs = "\n".join(["10", "Alice_X 1 2", "Alice_X 1 Bob_Y 1"])
    (root / "pairs_CPLFW.txt").write_text(pairs + "\n", encoding="utf-8")
    return root


def make_synthetic_agedb_flat(root: Path) -> Path:
    """Create AgeDB flat filename layout."""
    root.mkdir(parents=True, exist_ok=True)
    files = [
        "0_Maria_35_f.jpg",
        "1_Maria_70_f.jpg",
        "2_John_20_m.jpg",
        "3_John_55_m.jpg",
    ]
    for name in files:
        _touch_jpg(root / name)
    pairs = "\n".join(
        [
            "0_Maria_35_f.jpg 1_Maria_70_f.jpg 1",
            "0_Maria_35_f.jpg 2_John_20_m.jpg 0",
        ]
    )
    (root / "agedb_30_pairs.txt").write_text(pairs + "\n", encoding="utf-8")
    return root


def make_synthetic_tinyface(root: Path) -> Path:
    """Create TinyFace Gallery/Probe layout."""
    root.mkdir(parents=True, exist_ok=True)
    for identity in ("id_a", "id_b"):
        _touch_jpg(root / "Gallery" / identity / "0001.jpg")
        _touch_jpg(root / "Probe" / identity / "0001.jpg")
    return root


def make_synthetic_ar_face(root: Path) -> Path:
    """Create AR Face identity folders with occlusion-named files."""
    root.mkdir(parents=True, exist_ok=True)
    _touch_jpg(root / "m-001" / "neutral.jpg")
    _touch_jpg(root / "m-001" / "sunglasses.jpg")
    _touch_jpg(root / "w-001" / "neutral.jpg")
    _touch_jpg(root / "w-001" / "scarf.jpg")
    return root


def make_synthetic_chokepoint(root: Path) -> Path:
    """Create ChokePoint-like subject/camera folders."""
    root.mkdir(parents=True, exist_ok=True)
    _touch_jpg(root / "subject_001" / "cam_a" / "0001.jpg")
    _touch_jpg(root / "subject_001" / "cam_b" / "0001.jpg")
    _touch_jpg(root / "subject_002" / "cam_a" / "0001.jpg")
    _touch_jpg(root / "subject_002" / "cam_b" / "0001.jpg")
    return root


def make_synthetic_ytf(root: Path) -> Path:
    """Create YTF person/video/frame layout with pairs.txt."""
    root.mkdir(parents=True, exist_ok=True)
    for person, videos in (("Ann_Smith", ("1", "2")), ("Bob_Jones", ("1",))):
        for video in videos:
            _touch_jpg(root / person / video / "0.jpg")
            _touch_jpg(root / person / video / "1.jpg")
    pairs = "\n".join(
        [
            "Ann_Smith/1 Ann_Smith/2 1",
            "Ann_Smith/1 Bob_Jones/1 0",
        ]
    )
    (root / "pairs.txt").write_text(pairs + "\n", encoding="utf-8")
    return root
