"""AR Face Database adapter."""

from __future__ import annotations

from pathlib import Path

from facebench.datasets.base import BaseDataset
from facebench.datasets.common import (
    gallery_first_probe_rest,
    has_any_image,
    index_identity_directories,
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

_OCCLUSION_TOKENS = (
    "sunglass",
    "sunglasses",
    "scarf",
    "occlusion",
    "occluded",
)


class ARFaceDataset(BaseDataset):
    """Adapter for AR Face occlusion / illumination evaluation.

    Expected layout::

        <root>/
          m-001/
            *.bmp|*.jpg
          w-001/
            ...
          pairs.txt   # optional path_a path_b label
    """

    name = "AR-Face"
    category = "occlusion"
    prep_doc = "docs/datasets/ar_face.md"

    def __init__(
        self,
        root_path: str | Path,
        *,
        pairs_file: str | Path | None = None,
        image_extensions: set[str] | None = None,
    ) -> None:
        """Initialize the AR Face adapter.

        Args:
            root_path: Local AR Face root.
            pairs_file: Optional verification pair list.
            image_extensions: Allowed image suffixes.
        """
        super().__init__(root_path)
        self._pairs_override = Path(pairs_file) if pairs_file else None
        self._extensions = normalize_extensions(image_extensions)
        self._validator = IntegrityValidator()

    def load_dataset(self) -> DatasetIndex:
        """Index AR Face identity folders."""
        index = index_identity_directories(
            self.root_path,
            dataset_name=self.name,
            category=self.category,
            extensions=self._extensions,
            recursive_images=True,
            skip_dir_names={"protocol", "protocols", "meta"},
        )
        # Annotate likely occlusion/illumination cues from filenames.
        for samples in index.samples_by_identity.values():
            for sample in samples:
                lower = sample.path.name.lower()
                sample.metadata["occlusion_cue"] = any(
                    token in lower for token in _OCCLUSION_TOKENS
                )
        self._index = index
        return index

    def load_identity_pairs(self) -> list[IdentityPair]:
        """Load AR Face pairs from protocol file or synthesize within identities."""
        pairs_path = self.resolve_pairs_file()
        if pairs_path is not None:
            return parse_path_label_pairs(
                pairs_path,
                root_path=self.root_path,
                dataset_name=self.name,
            )
        return self._synthesize_pairs()

    def load_gallery(self) -> list[Sample]:
        """Return first image per identity as gallery."""
        gallery, _ = gallery_first_probe_rest(self.get_index(), self.preprocess)
        return gallery

    def load_probe(self) -> list[Sample]:
        """Return remaining images as probe."""
        _, probe = gallery_first_probe_rest(self.get_index(), self.preprocess)
        return probe

    def preprocess(self, sample: Sample) -> Sample:
        """Normalize AR Face sample paths and labels."""
        return preprocess_sample(
            sample, root_path=self.root_path, dataset_name=self.name
        )

    def validate_integrity(self) -> ValidationResult:
        """Validate AR Face root and image presence."""
        return self._validator.validate(
            self.root_path,
            require_subdirectories=True,
            prep_doc=self.prep_doc,
            extra_checks=[
                (
                    has_any_image(self.root_path, self._extensions),
                    f"No images found under {self.root_path}. See {self.prep_doc}",
                )
            ],
        )

    def resolve_pairs_file(self) -> Path | None:
        """Resolve an optional AR Face pairs protocol file."""
        return resolve_named_file(
            self.root_path,
            override=self._pairs_override,
            candidates=["pairs.txt", "ar_pairs.txt", "pair_list.txt"],
        )

    def _synthesize_pairs(self) -> list[IdentityPair]:
        """Build same/different pairs from multi-image identities."""
        index = self.get_index()
        pairs: list[IdentityPair] = []
        identities = index.identities
        for identity in identities:
            samples = index.samples_by_identity[identity]
            if len(samples) >= 2:
                pairs.append(
                    IdentityPair(
                        sample_a=self.preprocess(samples[0]),
                        sample_b=self.preprocess(samples[1]),
                        issame=True,
                        metadata={"synthetic": True},
                    )
                )
        for idx, identity in enumerate(identities[:-1]):
            other = identities[idx + 1]
            a = index.samples_by_identity[identity][0]
            b = index.samples_by_identity[other][0]
            pairs.append(
                IdentityPair(
                    sample_a=self.preprocess(a),
                    sample_b=self.preprocess(b),
                    issame=False,
                    metadata={"synthetic": True},
                )
            )
        if not pairs:
            raise FileNotFoundError(
                "AR Face pairs file not found and unable to synthesize pairs. "
                f"See {self.prep_doc}"
            )
        return pairs
