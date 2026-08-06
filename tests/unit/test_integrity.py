"""Unit tests for IntegrityValidator."""

from __future__ import annotations

from pathlib import Path

from facebench.datasets.integrity import IntegrityValidator


def test_validate_missing_root(tmp_path: Path) -> None:
    """Missing roots fail with prep_doc hints."""
    validator = IntegrityValidator()
    result = validator.validate(
        tmp_path / "missing",
        prep_doc="docs/datasets/lfw.md",
    )
    assert result.ok is False
    assert result.prep_doc == "docs/datasets/lfw.md"
    assert result.missing


def test_validate_required_paths_and_subdirs(tmp_path: Path) -> None:
    """Required relative paths and subdirectory counts are enforced."""
    root = tmp_path / "ds"
    root.mkdir()
    (root / "meta.txt").write_text("ok", encoding="utf-8")
    (root / "id1").mkdir()

    validator = IntegrityValidator()
    ok = validator.validate(
        root,
        required_paths=["meta.txt"],
        require_subdirectories=True,
        min_subdirectories=1,
    )
    assert ok.ok is True

    bad = validator.validate(
        root,
        required_paths=["pairs.txt"],
        require_subdirectories=True,
        min_subdirectories=2,
        prep_doc="docs/datasets/lfw.md",
    )
    assert bad.ok is False
    assert any("pairs.txt" in m for m in bad.missing)
