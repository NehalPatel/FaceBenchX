"""LFW (Labeled Faces in the Wild) dataset adapter."""

from __future__ import annotations

import re
from pathlib import Path

from facebench.datasets.base import BaseDataset
from facebench.datasets.integrity import IntegrityValidator
from facebench.datasets.types import (
    DatasetIndex,
    IdentityPair,
    Sample,
    ValidationResult,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
_PAIR_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class LFWDataset(BaseDataset):
    """Adapter for the LFW unrestricted / view-2 verification protocol.

    Expected layout (``root_path`` points at the identity folder tree)::

        <root_path>/
          Aaron_Eckhart/
            Aaron_Eckhart_0001.jpg
          ...
        pairs.txt   # beside or inside root_path (see ``pairs_file``)

    The standard ``pairs.txt`` format is supported:

    * Line 1: number of folds (typically ``10``) or pair-count header.
    * Following lines with 3 tokens: same-identity pair.
    * Following lines with 4 tokens: different-identity pair.
    """

    name = "LFW"
    category = "general"
    prep_doc = "docs/datasets/lfw.md"

    def __init__(
        self,
        root_path: str | Path,
        *,
        pairs_file: str | Path | None = None,
        image_extensions: set[str] | None = None,
    ) -> None:
        """Initialize the LFW adapter.

        Args:
            root_path: Path to the LFW identity directory tree.
            pairs_file: Optional explicit path to ``pairs.txt``. When
                omitted, FaceBench searches common locations.
            image_extensions: Allowed image suffixes (lowercase).
        """
        super().__init__(root_path)
        self._pairs_file_override = (
            Path(pairs_file).expanduser() if pairs_file is not None else None
        )
        self._image_extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (image_extensions or _IMAGE_EXTENSIONS)
        }
        self._validator = IntegrityValidator()

    def load_dataset(self) -> DatasetIndex:
        """Index identity folders and images under the LFW root.

        Returns:
            :class:`DatasetIndex` for all discovered identities.

        Raises:
            FileNotFoundError: If ``root_path`` does not exist.
            NotADirectoryError: If ``root_path`` is not a directory.
        """
        if not self.root_path.exists():
            raise FileNotFoundError(f"LFW root not found: {self.root_path}")
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"LFW root is not a directory: {self.root_path}")

        samples_by_identity: dict[str, list[Sample]] = {}
        for identity_dir in sorted(p for p in self.root_path.iterdir() if p.is_dir()):
            identity = identity_dir.name
            samples: list[Sample] = []
            for image_path in sorted(identity_dir.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in self._image_extensions:
                    continue
                samples.append(
                    Sample(
                        path=image_path.resolve(),
                        identity=identity,
                        image_id=image_path.stem,
                        metadata={"source": "lfw_index"},
                    )
                )
            if samples:
                samples_by_identity[identity] = samples

        identities = sorted(samples_by_identity.keys())
        image_count = sum(len(v) for v in samples_by_identity.values())
        index = DatasetIndex(
            root_path=self.root_path,
            identities=identities,
            samples_by_identity=samples_by_identity,
            image_count=image_count,
            metadata={"dataset": self.name, "category": self.category},
        )
        self._index = index
        return index

    def load_identity_pairs(self) -> list[IdentityPair]:
        """Parse LFW ``pairs.txt`` into verification pairs.

        Returns:
            List of :class:`IdentityPair` with ``issame`` labels.

        Raises:
            FileNotFoundError: If no pairs file can be resolved.
            ValueError: If a pairs line is malformed.
        """
        pairs_path = self.resolve_pairs_file()
        if pairs_path is None or not pairs_path.is_file():
            raise FileNotFoundError(
                "LFW pairs file not found. Place pairs.txt next to or inside "
                f"the dataset root ({self.root_path}), or pass pairs_file=. "
                f"See {self.prep_doc}"
            )

        pairs: list[IdentityPair] = []
        lines = pairs_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return pairs

        start_idx = 0
        header = lines[0].strip()
        ten_fold_protocol = header == "10"
        if header.isdigit():
            start_idx = 1

        for raw in lines[start_idx:]:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            fold: int | None
            if ten_fold_protocol:
                fold = len(pairs) // 600
            elif header.isdigit():
                fold = 0
            else:
                fold = None

            if len(tokens) == 3:
                name, idx_a, idx_b = tokens
                sample_a = self._sample_from_lfw_name(name, int(idx_a))
                sample_b = self._sample_from_lfw_name(name, int(idx_b))
                pairs.append(
                    IdentityPair(
                        sample_a=sample_a,
                        sample_b=sample_b,
                        issame=True,
                        fold=fold,
                        metadata={"pairs_file": str(pairs_path)},
                    )
                )
            elif len(tokens) == 4:
                name_a, idx_a, name_b, idx_b = tokens
                sample_a = self._sample_from_lfw_name(name_a, int(idx_a))
                sample_b = self._sample_from_lfw_name(name_b, int(idx_b))
                pairs.append(
                    IdentityPair(
                        sample_a=sample_a,
                        sample_b=sample_b,
                        issame=False,
                        fold=fold,
                        metadata={"pairs_file": str(pairs_path)},
                    )
                )
            else:
                raise ValueError(f"Malformed LFW pairs line: {raw!r}")

        return pairs

    def load_gallery(self) -> list[Sample]:
        """Build a simple gallery using the first image per identity.

        Returns:
            One :class:`Sample` per identity (first sorted image).
        """
        index = self.get_index()
        gallery: list[Sample] = []
        for identity in index.identities:
            samples = index.samples_by_identity[identity]
            if samples:
                gallery.append(self.preprocess(samples[0]))
        return gallery

    def load_probe(self) -> list[Sample]:
        """Build a probe set from non-gallery images.

        Returns:
            All images except the first image of each identity. When an
            identity has only one image, it is omitted from the probe set.
        """
        index = self.get_index()
        probe: list[Sample] = []
        for identity in index.identities:
            samples = index.samples_by_identity[identity]
            for sample in samples[1:]:
                probe.append(self.preprocess(sample))
        return probe

    def preprocess(self, sample: Sample) -> Sample:
        """Normalize sample paths and identity labels.

        Args:
            sample: Raw sample.

        Returns:
            Sample with resolved path and stripped identity label.
        """
        path = Path(sample.path).expanduser()
        if not path.is_absolute():
            path = (self.root_path / path).resolve()
        else:
            path = path.resolve()

        identity = sample.identity.strip()
        image_id = sample.image_id or path.stem
        metadata = dict(sample.metadata)
        metadata.setdefault("dataset", self.name)
        metadata["exists"] = path.is_file()
        return Sample(
            path=path,
            identity=identity,
            image_id=image_id,
            metadata=metadata,
        )

    def validate_integrity(self) -> ValidationResult:
        """Validate LFW root layout, image folders, and pairs file.

        Returns:
            :class:`ValidationResult` with preparation-doc hints on failure.
        """
        pairs_path = self.resolve_pairs_file()
        pairs_ok = pairs_path is not None and pairs_path.is_file()
        extra: list[tuple[bool, str]] = [
            (
                pairs_ok,
                "Missing pairs.txt (searched inside and beside the dataset root). "
                f"See {self.prep_doc}",
            )
        ]

        base = self._validator.validate(
            self.root_path,
            require_subdirectories=True,
            min_subdirectories=1,
            prep_doc=self.prep_doc,
            extra_checks=extra,
        )
        if not base.ok:
            return base

        # Spot-check that at least one image exists under an identity folder.
        has_image = False
        for identity_dir in self.root_path.iterdir():
            if not identity_dir.is_dir():
                continue
            for child in identity_dir.iterdir():
                if child.is_file() and child.suffix.lower() in self._image_extensions:
                    has_image = True
                    break
            if has_image:
                break

        image_check = self._validator.validate(
            self.root_path,
            prep_doc=self.prep_doc,
            extra_checks=[
                (
                    has_image,
                    "No images found under identity folders. "
                    f"Expected files such as Name/Name_0001.jpg. See {self.prep_doc}",
                )
            ],
        )
        return self._validator.merge(base, image_check)

    def resolve_pairs_file(self) -> Path | None:
        """Resolve the pairs protocol file path.

        Search order:

        1. Explicit ``pairs_file`` constructor argument.
        2. ``<root_path>/pairs.txt``
        3. ``<root_path>/../pairs.txt``
        4. ``<root_path>/pairsDevTest.txt`` then ``pairsDevTrain.txt``

        Returns:
            Resolved path if found, otherwise ``None``.
        """
        if self._pairs_file_override is not None:
            path = self._pairs_file_override
            return path.resolve() if path.exists() else path.expanduser().resolve()

        candidates = [
            self.root_path / "pairs.txt",
            self.root_path.parent / "pairs.txt",
            self.root_path / "pairsDevTest.txt",
            self.root_path / "pairsDevTrain.txt",
            self.root_path.parent / "pairsDevTest.txt",
            self.root_path.parent / "pairsDevTrain.txt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _sample_from_lfw_name(self, name: str, image_number: int) -> Sample:
        """Build a sample for an LFW ``Name`` + image index reference.

        Args:
            name: Identity folder / name token from pairs.txt.
            image_number: 1-based image index (zero-padded to 4 digits).

        Returns:
            :class:`Sample` pointing at the conventional LFW filename.

        Raises:
            ValueError: If ``name`` contains illegal characters.
        """
        if not _PAIR_NAME_RE.match(name):
            raise ValueError(f"Illegal LFW identity token in pairs file: {name!r}")
        filename = f"{name}_{image_number:04d}.jpg"
        path = (self.root_path / name / filename).resolve()
        return self.preprocess(
            Sample(
                path=path,
                identity=name,
                image_id=f"{name}_{image_number:04d}",
                metadata={"lfw_image_number": image_number},
            )
        )
