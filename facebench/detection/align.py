"""Shared face detection and alignment utilities.

Baseline A uses no shared aligner (or :class:`PassthroughAligner`).
Baseline B uses :class:`~facebench.detection.retinaface.RetinaFaceAligner`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from facebench.models.imaging import ImageInput, load_image_rgb


class FaceDetectionError(RuntimeError):
    """Raised when a shared detector finds no usable face."""


@dataclass(slots=True)
class AlignedFace:
    """Aligned (or passthrough) face crop ready for embedding.

    Attributes:
        image: RGB ``HxWx3 uint8`` array.
        source: Original path or array provenance label.
        metadata: Extensible alignment metadata.
    """

    image: np.ndarray
    source: str
    metadata: dict[str, Any]


@runtime_checkable
class BaseAligner(Protocol):
    """Protocol for shared face aligners."""

    name: str

    def align(self, image: ImageInput) -> AlignedFace:
        """Detect/align ``image`` and return an :class:`AlignedFace` crop."""


class PassthroughAligner:
    """Load RGB images without geometric alignment (controlled baseline)."""

    name = "passthrough"

    def align(self, image: ImageInput) -> AlignedFace:
        """Load an image and return it as an :class:`AlignedFace`.

        Args:
            image: Filesystem path or RGB ndarray.

        Returns:
            Passthrough aligned face.
        """
        array = load_image_rgb(image)
        if isinstance(image, (str, Path)):
            source = str(Path(image).expanduser().resolve())
        else:
            source = "ndarray"
        return AlignedFace(
            image=array,
            source=source,
            metadata={"aligner": self.name},
        )


def get_default_aligner() -> PassthroughAligner:
    """Return the default shared aligner used by optional hooks/tests."""
    return PassthroughAligner()


def as_image_transform(aligner: BaseAligner):
    """Wrap an aligner as an RGB→RGB callable for evaluation hooks.

    Args:
        aligner: Shared aligner instance.

    Returns:
        Callable ``(rgb: ndarray) -> ndarray`` returning the aligned crop.
    """

    def _transform(image_rgb: np.ndarray) -> np.ndarray:
        return aligner.align(image_rgb).image

    return _transform
