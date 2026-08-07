"""Factory for shared detection / alignment backends."""

from __future__ import annotations

from typing import Any

from facebench.detection.align import BaseAligner, PassthroughAligner
from facebench.detection.retinaface import RetinaFaceAligner


class AlignerFactoryError(ValueError):
    """Raised when an aligner cannot be constructed from config."""


_NONE_BACKENDS = frozenset({"", "none", "off", "disabled", "null"})
_PASSTHROUGH = frozenset({"passthrough", "identity"})
_RETINAFACE = frozenset({"retinaface", "scrfd", "insightface", "insightface_scrfd"})


def create_aligner(
    detection: dict[str, Any] | None,
    *,
    device: str = "cpu",
) -> BaseAligner | None:
    """Build a shared aligner from a normalized ``detection`` config block.

    Args:
        detection: Detection/alignment mapping (may be ``None``).
        device: FaceBench device string forwarded to heavy backends.

    Returns:
        ``None`` for Baseline A (no shared align), otherwise an aligner.

    Raises:
        AlignerFactoryError: If ``backend`` is unknown or misconfigured.
    """
    cfg = detection or {}
    backend = str(cfg.get("backend", "none")).strip().lower()

    if backend in _NONE_BACKENDS:
        return None

    if backend in _PASSTHROUGH:
        return PassthroughAligner()

    if backend in _RETINAFACE:
        det_size_raw = cfg.get("det_size", [640, 640])
        if (
            not isinstance(det_size_raw, (list, tuple))
            or len(det_size_raw) != 2
        ):
            raise AlignerFactoryError(
                "detection.det_size must be a [height, width] pair"
            )
        weights = cfg.get("weights_path")
        return RetinaFaceAligner(
            weights_path=weights,
            model_name=str(cfg.get("model_name", "buffalo_l")),
            det_size=(int(det_size_raw[0]), int(det_size_raw[1])),
            output_size=int(cfg.get("output_size", 112)),
            device=device,
            crop_mode=str(cfg.get("crop_mode", "bbox_margin")),
            bbox_margin=float(cfg.get("bbox_margin", 0.35)),
        )

    raise AlignerFactoryError(
        f"Unknown detection.backend {backend!r}. "
        "Supported: none, passthrough, retinaface"
    )
