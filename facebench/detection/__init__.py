"""Shared face detection and alignment utilities."""

from __future__ import annotations

from facebench.detection.align import (
    AlignedFace,
    PassthroughAligner,
    get_default_aligner,
)

__all__ = [
    "AlignedFace",
    "PassthroughAligner",
    "get_default_aligner",
]
