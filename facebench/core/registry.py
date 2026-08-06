"""Dataset category registry for public FaceBench benchmarks.

Maps research categories (general, pose, age, …) to supported public
dataset names. Custom and user datasets are intentionally unsupported.
"""

from __future__ import annotations

from copy import deepcopy


class CategoryRegistry:
    """Registry of public evaluation datasets organized by category.

    Attributes:
        _name_to_category: Mapping from canonical dataset name to category.
        _aliases: Alternate spellings mapped to canonical dataset names.
    """

    def __init__(
        self,
        name_to_category: dict[str, str],
        aliases: dict[str, str] | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            name_to_category: Canonical dataset name → research category.
            aliases: Optional alias → canonical dataset name mapping.
        """
        self._name_to_category = dict(name_to_category)
        self._aliases = dict(aliases or {})

    def canonicalize(self, name: str) -> str:
        """Resolve an alias or canonical name to the canonical dataset name.

        Args:
            name: Dataset name or known alias.

        Returns:
            Canonical dataset name.

        Raises:
            KeyError: If the name is not registered.
        """
        if name in self._name_to_category:
            return name
        if name in self._aliases:
            return self._aliases[name]
        raise KeyError(f"Unknown dataset: {name!r}")

    def is_supported(self, name: str) -> bool:
        """Return whether ``name`` refers to a registered public dataset.

        Args:
            name: Dataset name or alias.

        Returns:
            ``True`` if supported, otherwise ``False``.
        """
        try:
            self.canonicalize(name)
        except KeyError:
            return False
        return True

    def get_category(self, name: str) -> str:
        """Return the research category for a dataset.

        Args:
            name: Dataset name or alias.

        Returns:
            Category label (e.g. ``"pose"``, ``"age"``).

        Raises:
            KeyError: If the dataset is unknown.
        """
        canonical = self.canonicalize(name)
        return self._name_to_category[canonical]

    def list_all(self) -> list[str]:
        """Return all canonical dataset names in stable order.

        Returns:
            Sorted list of canonical dataset names.
        """
        return sorted(self._name_to_category.keys())

    def list_by_category(self, category: str) -> list[str]:
        """Return canonical dataset names belonging to ``category``.

        Args:
            category: Research category label.

        Returns:
            Sorted list of dataset names in the category.
        """
        return sorted(
            name for name, cat in self._name_to_category.items() if cat == category
        )

    def list_categories(self) -> list[str]:
        """Return all registered research categories.

        Returns:
            Sorted unique category labels.
        """
        return sorted(set(self._name_to_category.values()))

    def as_dict(self) -> dict[str, str]:
        """Return a deep copy of the canonical name → category mapping.

        Returns:
            Independent copy of the registry mapping.
        """
        return deepcopy(self._name_to_category)


def get_default_registry() -> CategoryRegistry:
    """Build the default v1 public-dataset category registry.

    Returns:
        Registry covering LFW, CFP-FP, CPLFW, AgeDB-30, TinyFace,
        AR Face, ChokePoint, and YTF.
    """
    name_to_category = {
        "LFW": "general",
        "CFP-FP": "pose",
        "CPLFW": "pose",
        "AgeDB-30": "age",
        "TinyFace": "low_resolution",
        "AR-Face": "occlusion",
        "ChokePoint": "surveillance",
        "YTF": "video",
    }
    aliases = {
        "lfw": "LFW",
        "cfp-fp": "CFP-FP",
        "cfp_fp": "CFP-FP",
        "CFP_FP": "CFP-FP",
        "cplfw": "CPLFW",
        "agedb-30": "AgeDB-30",
        "agedb": "AgeDB-30",
        "AgeDB": "AgeDB-30",
        "tinyface": "TinyFace",
        "AR Face": "AR-Face",
        "ar-face": "AR-Face",
        "ar_face": "AR-Face",
        "chokepoint": "ChokePoint",
        "ytf": "YTF",
        "YouTube Faces": "YTF",
    }
    return CategoryRegistry(name_to_category=name_to_category, aliases=aliases)
