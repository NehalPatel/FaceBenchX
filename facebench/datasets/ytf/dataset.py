"""YouTube Faces (YTF) dataset adapter."""

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


class YTFDataset(BaseDataset):
    """Adapter for YouTube Faces video verification.

    Expected layout::

        <root>/
          Person_Name/
            1/
              *.jpg          # frames
            2/
              *.jpg
          pairs.txt          # path_or_video_a path_or_video_b label
          splits.txt         # alternate protocol filename
    """

    name = "YTF"
    category = "video"
    prep_doc = "docs/datasets/ytf.md"

    def __init__(
        self,
        root_path: str | Path,
        *,
        pairs_file: str | Path | None = None,
        frame_aggregation: str = "first",
        image_extensions: set[str] | None = None,
    ) -> None:
        """Initialize the YTF adapter.

        Args:
            root_path: Local YTF root (person folders with video subfolders).
            pairs_file: Optional verification pair list.
            frame_aggregation: Strategy for representing a video clip in
                pair protocols that reference video directories:
                ``first`` (default) uses the first frame path.
            image_extensions: Allowed image suffixes.
        """
        super().__init__(root_path)
        if frame_aggregation not in {"first"}:
            raise ValueError(
                "Unsupported frame_aggregation "
                f"{frame_aggregation!r}; M3 supports 'first' only"
            )
        self._pairs_override = Path(pairs_file) if pairs_file else None
        self._frame_aggregation = frame_aggregation
        self._extensions = normalize_extensions(image_extensions)
        self._validator = IntegrityValidator()

    def load_dataset(self) -> DatasetIndex:
        """Index YTF person folders and nested video frame images."""
        index = index_identity_directories(
            self.root_path,
            dataset_name=self.name,
            category=self.category,
            extensions=self._extensions,
            recursive_images=True,
            skip_dir_names={
                "protocol",
                "protocols",
                "meta",
                "frame_images_db",
            },
        )
        # If root is frame_images_DB parent, also try that child.
        if index.image_count == 0:
            nested = self.root_path / "frame_images_DB"
            if nested.is_dir():
                index = index_identity_directories(
                    nested,
                    dataset_name=self.name,
                    category=self.category,
                    extensions=self._extensions,
                    recursive_images=True,
                )
                # Keep configured root_path; samples resolve via preprocess.
        for samples in index.samples_by_identity.values():
            for sample in samples:
                try:
                    rel = sample.path.relative_to(index.root_path)
                    if len(rel.parts) >= 2:
                        sample.metadata["video_id"] = rel.parts[1]
                except ValueError:
                    pass
        self._index = index
        return index

    def load_identity_pairs(self) -> list[IdentityPair]:
        """Load YTF verification pairs (frame or video-directory references)."""
        pairs_path = self.resolve_pairs_file()
        if pairs_path is None:
            return self._synthesize_pairs()
        # Prefer path/label parser; resolve video dirs to representative frames.
        raw_pairs = parse_path_label_pairs(
            pairs_path,
            root_path=self._effective_image_root(),
            dataset_name=self.name,
        )
        resolved: list[IdentityPair] = []
        for pair in raw_pairs:
            sample_a = self._resolve_video_or_frame(pair.sample_a)
            sample_b = self._resolve_video_or_frame(pair.sample_b)
            resolved.append(
                IdentityPair(
                    sample_a=sample_a,
                    sample_b=sample_b,
                    issame=pair.issame,
                    fold=pair.fold,
                    metadata=dict(pair.metadata),
                )
            )
        return resolved

    def load_gallery(self) -> list[Sample]:
        """Return first frame per identity as gallery."""
        gallery, _ = gallery_first_probe_rest(self.get_index(), self.preprocess)
        return gallery

    def load_probe(self) -> list[Sample]:
        """Return remaining frames as probe."""
        _, probe = gallery_first_probe_rest(self.get_index(), self.preprocess)
        return probe

    def preprocess(self, sample: Sample) -> Sample:
        """Normalize YTF sample paths and labels."""
        return preprocess_sample(
            sample,
            root_path=self._effective_image_root(),
            dataset_name=self.name,
        )

    def validate_integrity(self) -> ValidationResult:
        """Validate YTF root and frame image presence."""
        image_root = self._effective_image_root()
        return self._validator.validate(
            self.root_path,
            prep_doc=self.prep_doc,
            extra_checks=[
                (
                    has_any_image(image_root, self._extensions),
                    "No frame images found under the YTF root "
                    f"(or frame_images_DB/). See {self.prep_doc}",
                ),
                (
                    any(p.is_dir() for p in image_root.iterdir()),
                    "Expected person identity directories with video subfolders. "
                    f"See {self.prep_doc}",
                ),
            ],
        )

    def resolve_pairs_file(self) -> Path | None:
        """Resolve a YTF pairs / splits protocol file."""
        return resolve_named_file(
            self.root_path,
            override=self._pairs_override,
            candidates=[
                "pairs.txt",
                "splits.txt",
                "ytf_pairs.txt",
                "pair_list.txt",
            ],
        )

    def _effective_image_root(self) -> Path:
        """Return the directory that contains person folders."""
        nested = self.root_path / "frame_images_DB"
        if nested.is_dir():
            return nested.resolve()
        return self.root_path

    def _resolve_video_or_frame(self, sample: Sample) -> Sample:
        """If ``sample.path`` is a video directory, pick a representative frame."""
        path = sample.path
        if path.is_dir():
            frames = iter_image_files_recursive(path, self._extensions)
            if not frames:
                return self.preprocess(sample)
            frame = frames[0]
            return self.preprocess(
                Sample(
                    path=frame,
                    identity=sample.identity,
                    image_id=frame.stem,
                    metadata={
                        **sample.metadata,
                        "video_dir": str(path),
                        "frame_aggregation": self._frame_aggregation,
                    },
                )
            )
        return self.preprocess(sample)

    def _synthesize_pairs(self) -> list[IdentityPair]:
        """Synthesize pairs across videos of the same/different people."""
        index = self.get_index()
        pairs: list[IdentityPair] = []
        for identity in index.identities:
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
        identities = index.identities
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
                "YTF pairs file not found and unable to synthesize pairs. "
                f"See {self.prep_doc}"
            )
        return pairs
