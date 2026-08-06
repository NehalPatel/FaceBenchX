"""TinyFace low-resolution face recognition dataset adapter."""

from __future__ import annotations

from pathlib import Path

from facebench.datasets.base import BaseDataset
from facebench.datasets.common import (
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


class TinyFaceDataset(BaseDataset):
    """Adapter for TinyFace gallery/probe identification and optional pairs.

    Preferred layout::

        <root>/
          Gallery/
            identity_a/*.jpg
          Probe/
            identity_a/*.jpg
          pairs.txt                 # optional path_a path_b label
    """

    name = "TinyFace"
    category = "low_resolution"
    prep_doc = "docs/datasets/tinyface.md"

    def __init__(
        self,
        root_path: str | Path,
        *,
        pairs_file: str | Path | None = None,
        gallery_dirname: str = "Gallery",
        probe_dirname: str = "Probe",
        image_extensions: set[str] | None = None,
    ) -> None:
        """Initialize the TinyFace adapter.

        Args:
            root_path: Local TinyFace root.
            pairs_file: Optional verification pair list.
            gallery_dirname: Gallery subdirectory name.
            probe_dirname: Probe subdirectory name.
            image_extensions: Allowed image suffixes.
        """
        super().__init__(root_path)
        self._pairs_override = Path(pairs_file) if pairs_file else None
        self._gallery_dirname = gallery_dirname
        self._probe_dirname = probe_dirname
        self._extensions = normalize_extensions(image_extensions)
        self._validator = IntegrityValidator()

    def load_dataset(self) -> DatasetIndex:
        """Index gallery and probe identity images into one combined index."""
        gallery_root = self._split_root(self._gallery_dirname)
        probe_root = self._split_root(self._probe_dirname)
        samples_by_identity: dict[str, list[Sample]] = {}

        for split_name, split_root in (
            ("gallery", gallery_root),
            ("probe", probe_root),
        ):
            if split_root is None or not split_root.is_dir():
                continue
            for identity_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
                identity = identity_dir.name
                for image_path in iter_image_files_recursive(
                    identity_dir, self._extensions
                ):
                    sample = Sample(
                        path=image_path.resolve(),
                        identity=identity,
                        image_id=image_path.stem,
                        metadata={
                            "source": "tinyface_index",
                            "split": split_name,
                        },
                    )
                    samples_by_identity.setdefault(identity, []).append(sample)

        # Fallback: plain identity folders when Gallery/Probe are absent.
        if not samples_by_identity:
            index = index_identity_directories(
                self.root_path,
                dataset_name=self.name,
                category=self.category,
                extensions=self._extensions,
                recursive_images=True,
                skip_dir_names={"protocol", "protocols", "meta"},
            )
            self._index = index
            return index

        identities = sorted(samples_by_identity.keys())
        index = DatasetIndex(
            root_path=self.root_path,
            identities=identities,
            samples_by_identity=samples_by_identity,
            image_count=sum(len(v) for v in samples_by_identity.values()),
            metadata={
                "dataset": self.name,
                "category": self.category,
                "layout": "gallery_probe",
            },
        )
        self._index = index
        return index

    def load_identity_pairs(self) -> list[IdentityPair]:
        """Load optional TinyFace verification pairs, or synthesize same/diff.

        Returns:
            Explicit pairs from a protocol file when present; otherwise a
            small synthetic set derived from gallery/probe same-identity
            overlaps for smoke testing protocols.
        """
        pairs_path = self.resolve_pairs_file()
        if pairs_path is not None:
            return parse_path_label_pairs(
                pairs_path,
                root_path=self.root_path,
                dataset_name=self.name,
            )
        return self._synthesize_pairs_from_splits()

    def load_gallery(self) -> list[Sample]:
        """Load gallery samples from the Gallery split when available."""
        index = self.get_index()
        gallery: list[Sample] = []
        for identity in index.identities:
            for sample in index.samples_by_identity[identity]:
                if sample.metadata.get("split", "gallery") == "gallery":
                    gallery.append(self.preprocess(sample))
        if gallery:
            return gallery
        # Fallback: first image per identity.
        for identity in index.identities:
            samples = index.samples_by_identity[identity]
            if samples:
                gallery.append(self.preprocess(samples[0]))
        return gallery

    def load_probe(self) -> list[Sample]:
        """Load probe samples from the Probe split when available."""
        index = self.get_index()
        probe: list[Sample] = []
        for identity in index.identities:
            for sample in index.samples_by_identity[identity]:
                if sample.metadata.get("split") == "probe":
                    probe.append(self.preprocess(sample))
        if probe:
            return probe
        for identity in index.identities:
            for sample in index.samples_by_identity[identity][1:]:
                probe.append(self.preprocess(sample))
        return probe

    def preprocess(self, sample: Sample) -> Sample:
        """Normalize TinyFace sample paths and labels."""
        return preprocess_sample(
            sample, root_path=self.root_path, dataset_name=self.name
        )

    def validate_integrity(self) -> ValidationResult:
        """Validate TinyFace root and image presence."""
        gallery_root = self._split_root(self._gallery_dirname)
        probe_root = self._split_root(self._probe_dirname)
        has_splits = bool(
            gallery_root
            and gallery_root.is_dir()
            and probe_root
            and probe_root.is_dir()
        )
        return self._validator.validate(
            self.root_path,
            prep_doc=self.prep_doc,
            extra_checks=[
                (
                    has_any_image(self.root_path, self._extensions),
                    f"No images found under {self.root_path}. See {self.prep_doc}",
                ),
                (
                    has_splits or any(p.is_dir() for p in self.root_path.iterdir()),
                    "Expected Gallery/ and Probe/ directories (or identity folders). "
                    f"See {self.prep_doc}",
                ),
            ],
        )

    def resolve_pairs_file(self) -> Path | None:
        """Resolve an optional TinyFace pairs protocol file."""
        return resolve_named_file(
            self.root_path,
            override=self._pairs_override,
            candidates=["pairs.txt", "tinyface_pairs.txt", "pair_list.txt"],
        )

    def _split_root(self, dirname: str) -> Path | None:
        """Return a split directory if it exists (case-insensitive match)."""
        direct = self.root_path / dirname
        if direct.is_dir():
            return direct
        lowered = dirname.lower()
        for child in self.root_path.iterdir():
            if child.is_dir() and child.name.lower() == lowered:
                return child
        return None

    def _synthesize_pairs_from_splits(self) -> list[IdentityPair]:
        """Create a minimal same/different pair set from gallery/probe."""
        gallery = self.load_gallery()
        probe = self.load_probe()
        pairs: list[IdentityPair] = []
        by_id_gallery = {s.identity: s for s in gallery}
        by_id_probe: dict[str, Sample] = {}
        for sample in probe:
            by_id_probe.setdefault(sample.identity, sample)

        # Same-identity pairs where both splits contain the identity.
        for identity, g_sample in by_id_gallery.items():
            p_sample = by_id_probe.get(identity)
            if p_sample is not None:
                pairs.append(
                    IdentityPair(
                        sample_a=g_sample,
                        sample_b=p_sample,
                        issame=True,
                        metadata={"synthetic": True},
                    )
                )

        # Different-identity pairs from neighboring gallery/probe entries.
        identities = sorted(by_id_gallery.keys())
        for idx, identity in enumerate(identities[:-1]):
            other = identities[idx + 1]
            if other in by_id_probe:
                pairs.append(
                    IdentityPair(
                        sample_a=by_id_gallery[identity],
                        sample_b=by_id_probe[other],
                        issame=False,
                        metadata={"synthetic": True},
                    )
                )
        if not pairs:
            raise FileNotFoundError(
                "TinyFace pairs file not found and unable to synthesize pairs "
                f"from Gallery/Probe overlaps. See {self.prep_doc}"
            )
        return pairs
