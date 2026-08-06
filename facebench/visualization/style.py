"""Shared publication plotting style for FaceBench figures."""

from __future__ import annotations

from typing import Any


def apply_publication_style() -> dict[str, Any]:
    """Apply IEEE/Springer-friendly Matplotlib rc settings.

    Returns:
        The rc parameter dictionary that was applied.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    import matplotlib as mpl

    params: dict[str, Any] = {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    mpl.rcParams.update(params)
    return params


def use_agg_backend() -> None:
    """Force the non-interactive Agg backend for headless rendering."""
    import matplotlib

    matplotlib.use("Agg", force=True)
