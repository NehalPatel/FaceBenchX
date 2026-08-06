"""Shared helpers for public dataset adapters."""

from __future__ import annotations

from pathlib import Path

from facebench.datasets.types import DatasetIndex, IdentityPair, Sample

IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".ppm",
    ".pgm",
    ".tif",
    ".tiff",
}


def normalize_extensions(extensions: set[str] | None = None) -> set[str]:
    """Normalize image extension strings to lowercase dotted suffixes.

    Args:
        extensions: Optional custom extension set.

    Returns:
        Normalized extension set.
    """
    source = extensions or IMAGE_EXTENSIONS
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in source}


def ensure_dataset_root(root_path: Path, dataset_name: str) -> None:
    """Ensure a dataset root exists and is a directory.

    Args:
        root_path: Dataset root path.
        dataset_name: Human-readable dataset label for errors.

    Raises:
        FileNotFoundError: If the path does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    if not root_path.exists():
        raise FileNotFoundError(f"{dataset_name} root not found: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"{dataset_name} root is not a directory: {root_path}")


def iter_image_files(directory: Path, extensions: set[str]) -> list[Path]:
    """List image files directly under a directory (non-recursive).

    Args:
        directory: Directory to scan.
        extensions: Allowed suffixes.

    Returns:
        Sorted image paths.
    """
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )


def iter_image_files_recursive(directory: Path, extensions: set[str]) -> list[Path]:
    """List image files under a directory tree recursively.

    Args:
        directory: Root directory to scan.
        extensions: Allowed suffixes.

    Returns:
        Sorted image paths.
    """
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def index_identity_directories(
    root_path: Path,
    *,
    dataset_name: str,
    category: str,
    extensions: set[str],
    recursive_images: bool = False,
    skip_dir_names: set[str] | None = None,
) -> DatasetIndex:
    """Index one-folder-per-identity layouts.

    Args:
        root_path: Dataset root containing identity directories.
        dataset_name: Canonical dataset name.
        category: Research category label.
        extensions: Allowed image suffixes.
        recursive_images: When ``True``, collect images recursively inside
            each identity folder (useful for video / multi-camera trees).
        skip_dir_names: Directory names to ignore at the root level.

    Returns:
        Populated :class:`DatasetIndex`.
    """
    ensure_dataset_root(root_path, dataset_name)
    skip = {name.lower() for name in (skip_dir_names or set())}
    samples_by_identity: dict[str, list[Sample]] = {}

    for identity_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        if identity_dir.name.lower() in skip:
            continue
        identity = identity_dir.name
        image_paths = (
            iter_image_files_recursive(identity_dir, extensions)
            if recursive_images
            else iter_image_files(identity_dir, extensions)
        )
        samples = [
            Sample(
                path=image_path.resolve(),
                identity=identity,
                image_id=image_path.stem,
                metadata={
                    "source": f"{dataset_name.lower()}_index",
                    "relative_path": str(image_path.relative_to(root_path)),
                },
            )
            for image_path in image_paths
        ]
        if samples:
            samples_by_identity[identity] = samples

    identities = sorted(samples_by_identity.keys())
    return DatasetIndex(
        root_path=root_path,
        identities=identities,
        samples_by_identity=samples_by_identity,
        image_count=sum(len(v) for v in samples_by_identity.values()),
        metadata={"dataset": dataset_name, "category": category},
    )


def gallery_first_probe_rest(
    index: DatasetIndex, preprocess
) -> tuple[list[Sample], list[Sample]]:
    """Build gallery=first image / probe=remaining split.

    Args:
        index: Dataset index.
        preprocess: Callable applied to each sample.

    Returns:
        ``(gallery, probe)`` sample lists.
    """
    gallery: list[Sample] = []
    probe: list[Sample] = []
    for identity in index.identities:
        samples = index.samples_by_identity[identity]
        if not samples:
            continue
        gallery.append(preprocess(samples[0]))
        for sample in samples[1:]:
            probe.append(preprocess(sample))
    return gallery, probe


def preprocess_sample(
    sample: Sample,
    *,
    root_path: Path,
    dataset_name: str,
) -> Sample:
    """Normalize path/identity fields shared by most adapters.

    Args:
        sample: Raw sample.
        root_path: Dataset root for resolving relative paths.
        dataset_name: Canonical dataset name stored in metadata.

    Returns:
        Normalized :class:`Sample`.
    """
    path = Path(sample.path).expanduser()
    if not path.is_absolute():
        path = (root_path / path).resolve()
    else:
        path = path.resolve()
    metadata = dict(sample.metadata)
    metadata.setdefault("dataset", dataset_name)
    metadata["exists"] = path.is_file()
    return Sample(
        path=path,
        identity=sample.identity.strip(),
        image_id=sample.image_id or path.stem,
        metadata=metadata,
    )


def resolve_named_file(
    root_path: Path,
    *,
    override: Path | None,
    candidates: list[str],
) -> Path | None:
    """Resolve a protocol/metadata file from override or candidate names.

    Args:
        root_path: Dataset root.
        override: Optional explicit file path.
        candidates: Relative candidate filenames to search under ``root_path``
            and its parent.

    Returns:
        Resolved path if found, otherwise ``None``.
    """
    if override is not None:
        path = override.expanduser()
        return path.resolve()

    search_roots = [root_path, root_path.parent]
    for base in search_roots:
        for name in candidates:
            candidate = base / name
            if candidate.is_file():
                return candidate.resolve()
    return None


def parse_path_label_pairs(
    pairs_path: Path,
    *,
    root_path: Path,
    dataset_name: str,
    identity_from_path: bool = True,
) -> list[IdentityPair]:
    """Parse ``path_a path_b label`` verification lists.

    Args:
        pairs_path: Protocol file path.
        root_path: Dataset root for relative paths.
        dataset_name: Dataset name for metadata.
        identity_from_path: Derive identity from parent folder name when
            ``True``; otherwise use the file stem.

    Returns:
        Parsed identity pairs.

    Raises:
        ValueError: If a line is malformed.
    """
    pairs: list[IdentityPair] = []
    for raw in pairs_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.replace(",", " ").split()
        if len(tokens) < 3:
            raise ValueError(f"Malformed pair line in {pairs_path}: {raw!r}")
        path_a, path_b, label_token = tokens[0], tokens[1], tokens[2]
        issame = label_token.lower() in {"1", "true", "same", "positive", "p"}
        sample_a = _sample_from_relpath(
            path_a,
            root_path=root_path,
            dataset_name=dataset_name,
            identity_from_path=identity_from_path,
        )
        sample_b = _sample_from_relpath(
            path_b,
            root_path=root_path,
            dataset_name=dataset_name,
            identity_from_path=identity_from_path,
        )
        pairs.append(
            IdentityPair(
                sample_a=sample_a,
                sample_b=sample_b,
                issame=issame,
                metadata={"pairs_file": str(pairs_path)},
            )
        )
    return pairs


def parse_lfw_style_pairs(
    pairs_path: Path,
    *,
    root_path: Path,
    dataset_name: str,
    image_formatter=None,
) -> list[IdentityPair]:
    """Parse LFW-style 3/4-token pair protocols.

    Args:
        pairs_path: Protocol file.
        root_path: Identity-folder dataset root.
        dataset_name: Dataset name for metadata.
        image_formatter: Optional ``(name, index) -> Path`` callable.
            Defaults to ``root/name/name_XXXX.jpg``.

    Returns:
        Parsed identity pairs.
    """

    def default_formatter(name: str, index: int) -> Path:
        return root_path / name / f"{name}_{index:04d}.jpg"

    formatter = image_formatter or default_formatter
    pairs: list[IdentityPair] = []
    lines = pairs_path.read_text(encoding="utf-8").splitlines()
    start_idx = 0
    if lines and lines[0].strip().isdigit():
        start_idx = 1

    for raw in lines[start_idx:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) == 3:
            name, idx_a, idx_b = tokens
            sample_a = _sample_from_name_index(
                name, int(idx_a), root_path, dataset_name, formatter
            )
            sample_b = _sample_from_name_index(
                name, int(idx_b), root_path, dataset_name, formatter
            )
            issame = True
        elif len(tokens) == 4:
            name_a, idx_a, name_b, idx_b = tokens
            sample_a = _sample_from_name_index(
                name_a, int(idx_a), root_path, dataset_name, formatter
            )
            sample_b = _sample_from_name_index(
                name_b, int(idx_b), root_path, dataset_name, formatter
            )
            issame = False
        else:
            raise ValueError(f"Malformed LFW-style pair line: {raw!r}")
        pairs.append(
            IdentityPair(
                sample_a=sample_a,
                sample_b=sample_b,
                issame=issame,
                metadata={"pairs_file": str(pairs_path)},
            )
        )
    return pairs


def has_any_image(root_path: Path, extensions: set[str]) -> bool:
    """Return whether any image exists under ``root_path``.

    Args:
        root_path: Directory to scan recursively.
        extensions: Allowed suffixes.

    Returns:
        ``True`` if at least one image file is found.
    """
    if not root_path.is_dir():
        return False
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            return True
    return False


def _sample_from_relpath(
    rel: str,
    *,
    root_path: Path,
    dataset_name: str,
    identity_from_path: bool,
) -> Sample:
    path = Path(rel)
    if not path.is_absolute():
        path = (root_path / path).resolve()
    else:
        path = path.resolve()
    if identity_from_path and path.parent != root_path:
        # Prefer identity folder; for deeper trees use top folder under root.
        try:
            relative = path.relative_to(root_path)
            identity = relative.parts[0]
        except ValueError:
            identity = path.parent.name
    else:
        identity = path.stem.split("_")[0]
    return preprocess_sample(
        Sample(path=path, identity=identity, image_id=path.stem),
        root_path=root_path,
        dataset_name=dataset_name,
    )


def _sample_from_name_index(
    name: str,
    index: int,
    root_path: Path,
    dataset_name: str,
    formatter,
) -> Sample:
    path = Path(formatter(name, index)).resolve()
    return preprocess_sample(
        Sample(
            path=path,
            identity=name,
            image_id=f"{name}_{index:04d}",
            metadata={"image_number": index},
        ),
        root_path=root_path,
        dataset_name=dataset_name,
    )
