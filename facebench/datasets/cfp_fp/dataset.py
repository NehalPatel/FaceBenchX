"""CFP-FP (Celebrities in Frontal-Profile) dataset adapter."""

from __future__ import annotations

from pathlib import Path

from facebench.datasets.base import BaseDataset
from facebench.datasets.common import (
    gallery_first_probe_rest,
    has_any_image,
    index_identity_directories,
    iter_image_files_recursive,
    normalize_extensions,
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


class CFPFPDataset(BaseDataset):
    """Adapter for CFP frontal-profile verification.

    Supported layouts::

        <root>/
          001/
            frontal/*.jpg
            profile/*.jpg
          ...
          pairs_fp.txt                 # path_a path_b label
          Protocol/Pair_list_F.txt     # alternate protocol location
    """

    name = "CFP-FP"
    category = "pose"
    prep_doc = "docs/datasets/cfp_fp.md"

    def __init__(
        self,
        root_path: str | Path,
        *,
        pairs_file: str | Path | None = None,
        image_extensions: set[str] | None = None,
    ) -> None:
        """Initialize the CFP-FP adapter.

        Args:
            root_path: Local CFP dataset root.
            pairs_file: Optional explicit pair-list path.
            image_extensions: Allowed image suffixes.
        """
        super().__init__(root_path)
        self._pairs_override = Path(pairs_file) if pairs_file else None
        self._extensions = normalize_extensions(image_extensions)
        self._validator = IntegrityValidator()

    def load_dataset(self) -> DatasetIndex:
        """Index subject folders (including frontal/profile subtrees)."""
        index = index_identity_directories(
            self.root_path,
            dataset_name=self.name,
            category=self.category,
            extensions=self._extensions,
            recursive_images=True,
            skip_dir_names={"protocol", "protocols", "meta", "pair_lists"},
        )
        self._index = index
        return index

    def load_identity_pairs(self) -> list[IdentityPair]:
        """Load frontal-profile verification pairs from a path/label list."""
        pairs_path = self.resolve_pairs_file()
        if pairs_path is None:
            raise FileNotFoundError(
                "CFP-FP pairs file not found. Expected pairs_fp.txt or "
                f"Protocol/Pair_list_F.txt under {self.root_path}. See {self.prep_doc}"
            )
        pairs = parse_path_label_pairs(
            pairs_path,
            root_path=self.root_path,
            dataset_name=self.name,
        )
        return [
            IdentityPair(
                sample_a=self.preprocess(pair.sample_a),
                sample_b=self.preprocess(pair.sample_b),
                issame=pair.issame,
                fold=pair.fold,
                metadata=dict(pair.metadata),
            )
            for pair in pairs
        ]

    def load_gallery(self) -> list[Sample]:
        """Return first image per identity as gallery."""
        gallery, _ = gallery_first_probe_rest(self.get_index(), self.preprocess)
        return gallery

    def load_probe(self) -> list[Sample]:
        """Return remaining images as probe."""
        _, probe = gallery_first_probe_rest(self.get_index(), self.preprocess)
        return probe

    def preprocess(self, sample: Sample) -> Sample:
        """Normalize CFP sample paths and labels."""
        out = preprocess_sample(
            sample, root_path=self.root_path, dataset_name=self.name
        )
        # Tag pose folder when present (frontal/profile).
        try:
            rel = out.path.relative_to(self.root_path)
            if len(rel.parts) >= 2:
                out.metadata["pose_split"] = rel.parts[1].lower()
        except ValueError:
            pass
        return out

    def validate_integrity(self) -> ValidationResult:
        """Validate CFP root, images, and pair list."""
        pairs_path = self.resolve_pairs_file()
        base = self._validator.validate(
            self.root_path,
            require_subdirectories=True,
            prep_doc=self.prep_doc,
            extra_checks=[
                (
                    pairs_path is not None and pairs_path.is_file(),
                    "Missing CFP-FP pair list "
                    "(pairs_fp.txt / Protocol/Pair_list_F.txt). "
                    f"See {self.prep_doc}",
                ),
                (
                    has_any_image(self.root_path, self._extensions),
                    f"No images found under {self.root_path}. See {self.prep_doc}",
                ),
            ],
        )
        return base

    def resolve_pairs_file(self) -> Path | None:
        """Resolve the CFP pair-list protocol file."""
        return resolve_named_file(
            self.root_path,
            override=self._pairs_override,
            candidates=[
                "pairs_fp.txt",
                "pairs.txt",
                "Pair_list_F.txt",
                "Protocol/Pair_list_F.txt",
                "Protocols/Pair_list_F.txt",
                "protocol/Pair_list_F.txt",
            ],
        )

    def list_pose_images(self, identity: str, pose: str) -> list[Path]:
        """List images for an identity/pose subdirectory.

        Args:
            identity: Subject folder name.
            pose: Pose folder name (``frontal`` or ``profile``).

        Returns:
            Image paths under that pose folder.
        """
        return iter_image_files_recursive(
            self.root_path / identity / pose, self._extensions
        )
