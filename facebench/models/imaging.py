"""Image loading helpers for recognizer adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ImageInput = str | Path | np.ndarray


def load_image_rgb(image: ImageInput) -> np.ndarray:
    """Load an image as an ``uint8`` RGB ``H x W x 3`` array.

    Args:
        image: Filesystem path or an already-loaded ndarray.

    Returns:
        RGB image array.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: If the array shape is invalid.
        ImportError: If Pillow is required but not installed.
    """
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3/4 image array, got shape {array.shape}")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    path = Path(image).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to load image files. "
            "Install with: pip install 'facebench[models]'"
        ) from exc

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)
