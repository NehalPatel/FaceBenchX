"""Paper-local architecture constructors for AdaFace / MagFace."""

from __future__ import annotations

from .loaders import build_adaface_ir50, build_magface_iresnet50

__all__ = ["build_adaface_ir50", "build_magface_iresnet50"]
