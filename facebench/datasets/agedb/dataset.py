"""AgeDB-30 dataset adapter."""

from __future__ import annotations

import re
from pathlib import Path

from facebench.datasets.base import BaseDataset
from facebench.datasets.common import (
    gallery_first_probe_rest,
    has_any_image,
    index_identity_directories,
    normalize_extensions,
    parse_lfw_style_pairs,
    parse_path_label_pairs,
    preprocess_sample,
    resolve_named_file,
)
from facebench.datasets.integrity import IntegrityValidator
from facebench.datasets.types import (
    DatasetIndex,
    IdentityPair,
    Sample,
    ValidationResult,
)

# Common AgeDB flat filename: 0_MariaCallas_35_f.jpg
_FLAT_NAME_RE = re.compile(
    r"^(?P<id>\d+)_(?P<name>.+)_(?P<age>\d+)_(?P<gender>[mfMF])$"
)


class AgeDBDataset(BaseDataset):
    """Adapter for AgeDB-30 age-invariant verification.

    Supports either identity folders or AgeDB's flat filename layout::

        <root>/
          0_MariaCallas_35_f.jpg
          ...
          agedb_30_pairs.txt
    """

    name = "AgeDB-30"
    category = "age"
    prep_doc = "docs/datasets/agedb.md"

    def __init__(
        self,
        root_path: str | Path,
        *,
        pairs_file: str | Path | None = None,
        image_extensions: set[str] | None = None,
    ) -> None:
        """Initialize the AgeDB-30 adapter.

        Args:
            root_path: Local AgeDB root.
            pairs_file: Optional explicit pairs file.
            image_extensions: Allowed image suffixes.
        """
        super().__init__(root_path)
        self._pairs_override = Path(pairs_file) if pairs_file else None
        self._extensions = normalize_extensions(image_extensions)
        self._validator = IntegrityValidator()

    def load_dataset(self) -> DatasetIndex:
        """Index AgeDB images from folders or flat files."""
        # Prefer identity directories when present.
        identity_dirs = [p for p in self.root_path.iterdir() if p.is_dir()]
        if identity_dirs:
            index = index_identity_directories(
                self.root_path,
                dataset_name=self.name,
                category=self.category,
                extensions=self._extensions,
                skip_dir_names={"protocol", "protocols", "meta"},
            )
            self._index = index
            return index

        samples_by_identity: dict[str, list[Sample]] = {}
        for image_path in sorted(self.root_path.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in self._extensions:
                continue
            identity, meta = self._parse_flat_identity(image_path.stem)
            sample = Sample(
                path=image_path.resolve(),
                identity=identity,
                image_id=image_path.stem,
                metadata={"source": "agedb_flat", **meta},
            )
            samples_by_identity.setdefault(identity, []).append(sample)

        identities = sorted(samples_by_identity.keys())
        index = DatasetIndex(
            root_path=self.root_path,
            identities=identities,
            samples_by_identity=samples_by_identity,
            image_count=sum(len(v) for v in samples_by_identity.values()),
            metadata={
                "dataset": self.name,
                "category": self.category,
                "layout": "flat",
            },
        )
        self._index = index
        return index

    def load_identity_pairs(self) -> list[IdentityPair]:
        """Load AgeDB-30 verification pairs."""
        pairs_path = self.resolve_pairs_file()
        if pairs_path is None:
            raise FileNotFoundError(
                "AgeDB pairs file not found "
                f"(agedb_30_pairs.txt / pairs.txt) under {self.root_path}. "
                f"See {self.prep_doc}"
            )
        for raw in pairs_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.isdigit():
                continue
            tokens = line.replace(",", " ").split()
            if len(tokens) >= 3 and (
                "/" in tokens[0] or "\\" in tokens[0] or "." in tokens[0]
            ):
                return parse_path_label_pairs(
                    pairs_path,
                    root_path=self.root_path,
                    dataset_name=self.name,
                    identity_from_path=False,
                )
            break
        return parse_lfw_style_pairs(
            pairs_path,
            root_path=self.root_path,
            dataset_name=self.name,
            image_formatter=self._format_agedb_image,
        )

    def load_gallery(self) -> list[Sample]:
        """Return first image per identity as gallery."""
        gallery, _ = gallery_first_probe_rest(self.get_index(), self.preprocess)
        return gallery

    def load_probe(self) -> list[Sample]:
        """Return remaining images as probe."""
        _, probe = gallery_first_probe_rest(self.get_index(), self.preprocess)
        return probe

    def preprocess(self, sample: Sample) -> Sample:
        """Normalize AgeDB sample paths and labels."""
        return preprocess_sample(
            sample, root_path=self.root_path, dataset_name=self.name
        )

    def validate_integrity(self) -> ValidationResult:
        """Validate AgeDB root, images, and pairs file."""
        pairs_path = self.resolve_pairs_file()
        return self._validator.validate(
            self.root_path,
            prep_doc=self.prep_doc,
            extra_checks=[
                (
                    pairs_path is not None and pairs_path.is_file(),
                    f"Missing AgeDB pairs file. See {self.prep_doc}",
                ),
                (
                    has_any_image(self.root_path, self._extensions),
                    f"No images found under {self.root_path}. See {self.prep_doc}",
                ),
            ],
        )

    def resolve_pairs_file(self) -> Path | None:
        """Resolve the AgeDB-30 pairs protocol file."""
        return resolve_named_file(
            self.root_path,
            override=self._pairs_override,
            candidates=[
                "agedb_30_pairs.txt",
                "AgeDB_30_pairs.txt",
                "pairs_agedb.txt",
                "pairs.txt",
            ],
        )

    def _format_agedb_image(self, name: str, index: int) -> Path:
        """Map LFW-style name/index references onto AgeDB folders when present."""
        folder = self.root_path / name
        if folder.is_dir():
            return folder / f"{name}_{index:04d}.jpg"
        # Flat layout fallback: keep a conventional Name_XXXX.jpg path.
        return self.root_path / f"{name}_{index:04d}.jpg"

    @staticmethod
    def _parse_flat_identity(stem: str) -> tuple[str, dict[str, str]]:
        """Parse AgeDB flat filename stem into identity + metadata."""
        match = _FLAT_NAME_RE.match(stem)
        if not match:
            return stem, {}
        return match.group("name"), {
            "agedb_id": match.group("id"),
            "age": match.group("age"),
            "gender": match.group("gender").lower(),
        }
