"""Helpers for building tiny synthetic LFW-like fixtures (not real faces)."""

from __future__ import annotations

from pathlib import Path


def make_synthetic_lfw(
    root: Path,
    *,
    pairs_beside_root: bool = False,
    include_pairs: bool = True,
) -> Path:
    """Create a minimal LFW-like directory tree for unit tests.

    Args:
        root: Directory that will contain identity folders (LFW root).
        pairs_beside_root: When ``True``, write ``pairs.txt`` in the parent
            of ``root``; otherwise write it inside ``root``.
        include_pairs: When ``False``, omit the pairs file entirely.

    Returns:
        The LFW root path (same as ``root``).
    """
    root.mkdir(parents=True, exist_ok=True)

    people = {
        "Alice_Example": [1, 2, 3],
        "Bob_Example": [1, 2],
        "Carol_Example": [1, 2],
    }
    for name, indices in people.items():
        person_dir = root / name
        person_dir.mkdir(parents=True, exist_ok=True)
        for idx in indices:
            # Empty placeholder files — not real face images.
            (person_dir / f"{name}_{idx:04d}.jpg").write_bytes(b"")

    if include_pairs:
        pairs_lines = [
            "10",
            "Alice_Example 1 2",
            "Bob_Example 1 2",
            "Alice_Example 1 Bob_Example 1",
            "Carol_Example 2 Alice_Example 3",
        ]
        # Pad remaining fold slots are unnecessary for unit tests; header
        # may be "10" while containing fewer lines — parser still works.
        pairs_text = "\n".join(pairs_lines) + "\n"
        if pairs_beside_root:
            (root.parent / "pairs.txt").write_text(pairs_text, encoding="utf-8")
        else:
            (root / "pairs.txt").write_text(pairs_text, encoding="utf-8")

    return root
