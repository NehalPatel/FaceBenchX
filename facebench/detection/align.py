"""Shared face detection and alignment utilities.

Phase 9 provides a minimal shared RGB load/align stage so models and
synthetic robustness transforms operate on a consistent image view.
Heavy detector backends can replace :class:`PassthroughAligner` later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from facebench.models.imaging import ImageInput, load_image_rgb


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
        return AlignedFace(image=array, source=source, metadata={"aligner": self.name})


def get_default_aligner() -> PassthroughAligner:
    """Return the default shared aligner used by benchmark execution."""
    return PassthroughAligner()
