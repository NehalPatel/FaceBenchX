"""CPLFW (Cross-Pose LFW) dataset adapter."""

from __future__ import annotations

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


class CPLFWDataset(BaseDataset):
    """Adapter for Cross-Pose LFW verification.

    Expected layout mirrors LFW identity folders with a CPLFW pair list::

        <root>/
          Name/
            Name_0001.jpg
          pairs_CPLFW.txt   # LFW-style or path/label lines
    """

    name = "CPLFW"
    category = "pose"
    prep_doc = "docs/datasets/cplfw.md"

    def __init__(
        self,
        root_path: str | Path,
        *,
        pairs_file: str | Path | None = None,
        image_extensions: set[str] | None = None,
    ) -> None:
        """Initialize the CPLFW adapter.

        Args:
            root_path: Local CPLFW identity tree.
            pairs_file: Optional explicit pairs file.
            image_extensions: Allowed image suffixes.
        """
        super().__init__(root_path)
        self._pairs_override = Path(pairs_file) if pairs_file else None
        self._extensions = normalize_extensions(image_extensions)
        self._validator = IntegrityValidator()

    def load_dataset(self) -> DatasetIndex:
        """Index CPLFW identity folders."""
        index = index_identity_directories(
            self.root_path,
            dataset_name=self.name,
            category=self.category,
            extensions=self._extensions,
        )
        self._index = index
        return index

    def load_identity_pairs(self) -> list[IdentityPair]:
        """Load CPLFW pairs (LFW-style or path/label)."""
        pairs_path = self.resolve_pairs_file()
        if pairs_path is None:
            raise FileNotFoundError(
                "CPLFW pairs file not found "
                f"(pairs_CPLFW.txt / pairs.txt) under {self.root_path}. "
                f"See {self.prep_doc}"
            )
        # Sniff format from the first data line.
        for raw in pairs_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.isdigit():
                continue
            tokens = line.replace(",", " ").split()
            if len(tokens) >= 3 and not tokens[1].isdigit():
                return parse_path_label_pairs(
                    pairs_path,
                    root_path=self.root_path,
                    dataset_name=self.name,
                )
            break
        return parse_lfw_style_pairs(
            pairs_path,
            root_path=self.root_path,
            dataset_name=self.name,
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
        """Normalize CPLFW sample paths and labels."""
        return preprocess_sample(
            sample, root_path=self.root_path, dataset_name=self.name
        )

    def validate_integrity(self) -> ValidationResult:
        """Validate CPLFW root, images, and pairs file."""
        pairs_path = self.resolve_pairs_file()
        return self._validator.validate(
            self.root_path,
            require_subdirectories=True,
            prep_doc=self.prep_doc,
            extra_checks=[
                (
                    pairs_path is not None and pairs_path.is_file(),
                    f"Missing CPLFW pairs file. See {self.prep_doc}",
                ),
                (
                    has_any_image(self.root_path, self._extensions),
                    f"No images found under {self.root_path}. See {self.prep_doc}",
                ),
            ],
        )

    def resolve_pairs_file(self) -> Path | None:
        """Resolve the CPLFW pairs protocol file."""
        return resolve_named_file(
            self.root_path,
            override=self._pairs_override,
            candidates=[
                "pairs_CPLFW.txt",
                "pairs_cplfw.txt",
                "pair_CPLFW.txt",
                "pairs.txt",
            ],
        )
