"""Category registry bridge for the datasets package.

Re-exports the core :class:`~facebench.core.registry.CategoryRegistry`
and adds preparation-doc lookups used by integrity validation.
"""

from __future__ import annotations

from facebench.core.registry import CategoryRegistry, get_default_registry

# Canonical dataset name → preparation guide path.
DATASET_PREP_DOCS: dict[str, str] = {
    "LFW": "docs/datasets/lfw.md",
    "CFP-FP": "docs/datasets/cfp_fp.md",
    "CPLFW": "docs/datasets/cplfw.md",
    "AgeDB-30": "docs/datasets/agedb.md",
    "TinyFace": "docs/datasets/tinyface.md",
    "AR-Face": "docs/datasets/ar_face.md",
    "ChokePoint": "docs/datasets/chokepoint.md",
    "YTF": "docs/datasets/ytf.md",
}


def get_prep_doc(dataset_name: str, registry: CategoryRegistry | None = None) -> str:
    """Return the preparation guide path for a public dataset.

    Args:
        dataset_name: Canonical name or alias.
        registry: Optional registry used for canonicalization.

    Returns:
        Relative documentation path.

    Raises:
        KeyError: If the dataset is unknown.
    """
    active = registry or get_default_registry()
    canonical = active.canonicalize(dataset_name)
    return DATASET_PREP_DOCS.get(canonical, "docs/datasets/README.md")


__all__ = [
    "DATASET_PREP_DOCS",
    "CategoryRegistry",
    "get_default_registry",
    "get_prep_doc",
]
