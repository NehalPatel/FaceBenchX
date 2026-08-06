"""Shared preprocessing helpers for ArcFace-family recognizers."""

from __future__ import annotations

from typing import Any

import numpy as np

from facebench.models.similarity import l2_normalize


def preprocess_arcface_112(
    image_rgb: np.ndarray,
    *,
    device: str = "cpu",
) -> Any:
    """Preprocess an RGB image to a ``1x3x112x112`` ArcFace tensor.

    Args:
        image_rgb: ``H x W x 3`` uint8 image.
        device: Torch device string.

    Returns:
        Torch tensor batch normalized to approximately ``[-1, 1]``.

    Raises:
        ImportError: If torch / Pillow / torchvision are unavailable.
    """
    from PIL import Image
    from torchvision import transforms

    image = Image.fromarray(image_rgb)
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)


def embedding_from_torch(output: Any) -> np.ndarray:
    """Convert a torch embedding output to an L2-normalized numpy vector.

    Args:
        output: Torch tensor (``1 x D`` or ``D``).

    Returns:
        1-D float64 embedding.
    """
    vector = output.detach().cpu().numpy().reshape(-1).astype(np.float64)
    return l2_normalize(vector)


def resolve_device_ctx_id(device: str) -> int:
    """Map a FaceBench device string to InsightFace ``ctx_id``.

    Args:
        device: ``cpu``, ``cuda``, or ``cuda:N``.

    Returns:
        ``-1`` for CPU, otherwise the CUDA device index.
    """
    normalized = device.strip().lower()
    if normalized == "cpu":
        return -1
    if normalized.startswith("cuda"):
        if ":" in normalized:
            return int(normalized.split(":", 1)[1])
        return 0
    return -1
