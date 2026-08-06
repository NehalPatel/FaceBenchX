"""Unit tests for the public dataset category registry."""

from __future__ import annotations

import pytest

from facebench.core.registry import get_default_registry


def test_default_registry_lists_eight_datasets() -> None:
    """v1 registry exposes exactly eight public datasets."""
    registry = get_default_registry()
    assert len(registry.list_all()) == 8


def test_category_lookup_and_aliases() -> None:
    """Canonical names and aliases resolve to the same category."""
    registry = get_default_registry()
    assert registry.get_category("LFW") == "general"
    assert registry.get_category("cfp_fp") == "pose"
    assert registry.canonicalize("AgeDB") == "AgeDB-30"
    assert registry.list_by_category("video") == ["YTF"]


def test_unknown_dataset_raises() -> None:
    """Unknown datasets raise KeyError."""
    registry = get_default_registry()
    with pytest.raises(KeyError):
        registry.get_category("CustomDataset")
