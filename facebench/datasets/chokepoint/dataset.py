"""ChokePoint surveillance dataset adapter."""

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


class ChokePointDataset(BaseDataset):
    """Adapter for ChokePoint multi-camera surveillance evaluation.

    Expected layout::

        <root>/
          subject_id/                 # or session/camera folders as identities
            *.jpg
          pairs.txt                   # optional path_a path_b label
          gallery.txt / probe.txt     # optional path lists
    """

    name = "ChokePoint"
    category = "surveillance"
    prep_doc = "docs/datasets/chokepoint.md"

    def __init__(
        self,
        root_path: str | Path,
        *,
        pairs_file: str | Path | None = None,
        gallery_list: str | Path | None = None,
        probe_list: str | Path | None = None,
        image_extensions: set[str] | None = None,
    ) -> None:
        """Initialize the ChokePoint adapter.

        Args:
            root_path: Local ChokePoint root.
            pairs_file: Optional verification pair list.
            gallery_list: Optional text file listing gallery image paths.
            probe_list: Optional text file listing probe image paths.
            image_extensions: Allowed image suffixes.
        """
        super().__init__(root_path)
        self._pairs_override = Path(pairs_file) if pairs_file else None
        self._gallery_list = Path(gallery_list) if gallery_list else None
        self._probe_list = Path(probe_list) if probe_list else None
        self._extensions = normalize_extensions(image_extensions)
        self._validator = IntegrityValidator()

    def load_dataset(self) -> DatasetIndex:
        """Index ChokePoint identity / session folders recursively."""
        index = index_identity_directories(
            self.root_path,
            dataset_name=self.name,
            category=self.category,
            extensions=self._extensions,
            recursive_images=True,
            skip_dir_names={"protocol", "protocols", "meta", "lists"},
        )
        for samples in index.samples_by_identity.values():
            for sample in samples:
                sample.metadata["camera_or_session"] = sample.path.parent.name
        self._index = index
        return index

    def load_identity_pairs(self) -> list[IdentityPair]:
        """Load ChokePoint pairs from protocol file or synthesize."""
        pairs_path = self.resolve_pairs_file()
        if pairs_path is not None:
            return parse_path_label_pairs(
                pairs_path,
                root_path=self.root_path,
                dataset_name=self.name,
            )
        return self._synthesize_pairs()

    def load_gallery(self) -> list[Sample]:
        """Load gallery from list file or first-image fallback."""
        listed = self._load_split_list("gallery")
        if listed:
            return listed
        gallery, _ = gallery_first_probe_rest(self.get_index(), self.preprocess)
        return gallery

    def load_probe(self) -> list[Sample]:
        """Load probe from list file or remaining-image fallback."""
        listed = self._load_split_list("probe")
        if listed:
            return listed
        _, probe = gallery_first_probe_rest(self.get_index(), self.preprocess)
        return probe

    def preprocess(self, sample: Sample) -> Sample:
        """Normalize ChokePoint sample paths and labels."""
        return preprocess_sample(
            sample, root_path=self.root_path, dataset_name=self.name
        )

    def validate_integrity(self) -> ValidationResult:
        """Validate ChokePoint root and image presence."""
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
        """Resolve an optional ChokePoint pairs protocol file."""
        return resolve_named_file(
            self.root_path,
            override=self._pairs_override,
            candidates=["pairs.txt", "chokepoint_pairs.txt", "pair_list.txt"],
        )

    def _load_split_list(self, split: str) -> list[Sample]:
        """Load gallery/probe samples from an optional path list file."""
        override = self._gallery_list if split == "gallery" else self._probe_list
        resolved = resolve_named_file(
            self.root_path,
            override=override,
            candidates=[f"{split}.txt", f"{split}_list.txt", f"lists/{split}.txt"],
        )
        if resolved is None:
            return []
        samples: list[Sample] = []
        for raw in resolved.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line.split()[0])
            if not path.is_absolute():
                path = (self.root_path / path).resolve()
            try:
                identity = path.relative_to(self.root_path).parts[0]
            except ValueError:
                identity = path.parent.name
            samples.append(
                self.preprocess(
                    Sample(
                        path=path,
                        identity=identity,
                        image_id=path.stem,
                        metadata={"split": split, "list_file": str(resolved)},
                    )
                )
            )
        return samples

    def _synthesize_pairs(self) -> list[IdentityPair]:
        """Synthesize same/different pairs from multi-image identities."""
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
            pairs.append(
                IdentityPair(
                    sample_a=self.preprocess(index.samples_by_identity[identity][0]),
                    sample_b=self.preprocess(index.samples_by_identity[other][0]),
                    issame=False,
                    metadata={"synthetic": True},
                )
            )
        if not pairs:
            raise FileNotFoundError(
                "ChokePoint pairs file not found and unable to synthesize pairs. "
                f"See {self.prep_doc}"
            )
        return pairs
