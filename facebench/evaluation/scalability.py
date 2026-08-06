"""Scalability ladders over public-gallery identity subsets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from facebench.datasets.base import BaseDataset
from facebench.datasets.types import Sample
from facebench.evaluation.identification import run_identification
from facebench.evaluation.types import ScalabilityPoint, ScalabilityReport
from facebench.matcher.base import BaseMatcher
from facebench.metrics.computational import ComputeProfiler

DEFAULT_IDENTITY_COUNTS: tuple[int, ...] = (10, 100, 500, 1000, 5000)


def subset_by_identities(
    gallery: list[Sample],
    probe: list[Sample],
    identity_count: int,
    *,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], str | None]:
    """Subset gallery/probe to at most ``identity_count`` shared identities.

    Args:
        gallery: Full gallery samples.
        probe: Full probe samples.
        identity_count: Target enrolled identity count.
        seed: RNG seed for reproducible subset selection.

    Returns:
        ``(gallery_subset, probe_subset, skip_reason)``. When the dataset
        cannot support ``identity_count``, subsets are empty and
        ``skip_reason`` explains why.
    """
    if identity_count < 1:
        return [], [], "identity_count must be >= 1"

    gallery_ids = sorted({sample.identity for sample in gallery})
    available = len(gallery_ids)
    if available < identity_count:
        return (
            [],
            [],
            f"dataset has {available} gallery identities; need {identity_count}",
        )

    rng = np.random.default_rng(seed)
    chosen = set(
        rng.choice(
            np.asarray(gallery_ids, dtype=object),
            size=identity_count,
            replace=False,
        ).tolist()
    )
    gallery_sub = [sample for sample in gallery if sample.identity in chosen]
    probe_sub = [sample for sample in probe if sample.identity in chosen]
    if not probe_sub:
        return (
            gallery_sub,
            [],
            f"no probe samples remain for {identity_count} selected identities",
        )
    return gallery_sub, probe_sub, None


def run_scalability_ladder(
    dataset: BaseDataset,
    model: Any,
    matcher: BaseMatcher,
    *,
    identity_counts: Sequence[int] | None = None,
    seed: int = 42,
    ranks: tuple[int, ...] = (1, 5, 10),
    profiler: ComputeProfiler | None = None,
) -> ScalabilityReport:
    """Run identification at each configured gallery identity count.

    Args:
        dataset: Public dataset adapter providing gallery/probe splits.
        model: Loaded recognizer.
        matcher: Similarity matcher.
        identity_counts: Ladder sizes (defaults to design set).
        seed: Subset selection seed.
        ranks: CMC ranks forwarded to identification.
        profiler: Optional shared profiler.

    Returns:
        :class:`ScalabilityReport`.
    """
    counts = list(identity_counts or DEFAULT_IDENTITY_COUNTS)
    gallery = dataset.load_gallery()
    probe = dataset.load_probe()
    points: list[ScalabilityPoint] = []

    for count in counts:
        gallery_sub, probe_sub, skip = subset_by_identities(
            gallery, probe, int(count), seed=seed
        )
        if skip is not None:
            points.append(
                ScalabilityPoint(
                    identity_count=int(count),
                    identification=None,
                    skipped_reason=skip,
                )
            )
            continue
        result = run_identification(
            gallery_sub,
            probe_sub,
            model,
            matcher,
            profiler,
            ranks=ranks,
        )
        points.append(
            ScalabilityPoint(
                identity_count=int(count),
                identification=result,
                skipped_reason=None,
            )
        )

    return ScalabilityReport(
        points=points,
        dataset_name=dataset.name,
        seed=seed,
    )
