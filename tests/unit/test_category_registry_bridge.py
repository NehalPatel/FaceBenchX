"""Tests for datasets category registry bridge."""

from __future__ import annotations

from facebench.datasets.category_registry import get_default_registry, get_prep_doc


def test_prep_doc_for_lfw() -> None:
    """LFW maps to the LFW preparation guide."""
    assert get_prep_doc("LFW") == "docs/datasets/lfw.md"
    assert get_prep_doc("lfw") == "docs/datasets/lfw.md"


def test_registry_reexport_lists_public_sets() -> None:
    """Bridge exposes the same eight public datasets as core."""
    registry = get_default_registry()
    assert len(registry.list_all()) == 8
