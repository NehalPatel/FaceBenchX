"""Publication-ready figure generation (Phase 7)."""

from __future__ import annotations

from facebench.visualization.figures import FigureGenerator
from facebench.visualization.style import apply_publication_style, use_agg_backend

__all__ = [
    "FigureGenerator",
    "apply_publication_style",
    "use_agg_backend",
]
