"""Identification (1:N) benchmark protocol."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from facebench.datasets.types import Sample
from facebench.evaluation.types import IdentificationResult
from facebench.matcher.base import BaseMatcher
from facebench.metrics.computational import ComputeProfiler
from facebench.models.imaging import load_image_rgb

ImageTransform = Callable[[np.ndarray], np.ndarray]


def run_identification(
    gallery: list[Sample],
    probe: list[Sample],
    model: Any,
    matcher: BaseMatcher,
    profiler: ComputeProfiler | None = None,
    *,
    ranks: tuple[int, ...] = (1, 5, 10),
    image_transform: ImageTransform | None = None,
) -> IdentificationResult:
    """Embed gallery/probe sets and compute rank-k identification metrics.

    Args:
        gallery: Enrollment samples (typically one image per identity).
        probe: Query samples with known identities.
        model: Recognizer exposing ``generate_embedding``.
        matcher: Similarity matcher (higher score = more similar).
        profiler: Optional profiler; embedding calls are tracked when set.
        ranks: Rank cutoffs for the cumulative match characteristic (CMC).
        image_transform: Optional RGB ndarray transform (Baseline B align).

    Returns:
        :class:`IdentificationResult`.

    Raises:
        ValueError: If gallery or probe is empty.
    """
    if not gallery:
        raise ValueError("Identification requires a non-empty gallery")
    if not probe:
        raise ValueError("Identification requires a non-empty probe set")

    gallery_embs = [
        _timed_embed(model, sample.path, profiler, image_transform)
        for sample in gallery
    ]
    gallery_ids = [sample.identity for sample in gallery]
    identities = sorted({sample.identity for sample in gallery})

    hits_at: dict[int, int] = {rank: 0 for rank in ranks}
    latencies: list[float] = []
    evaluated = 0

    for sample in probe:
        start = time.perf_counter()
        query = _timed_embed(model, sample.path, profiler, image_transform)
        scores = np.asarray(
            [float(matcher.score(query, emb)) for emb in gallery_embs],
            dtype=np.float64,
        )
        order = np.argsort(-scores)  # descending similarity
        ranked_ids = [gallery_ids[int(idx)] for idx in order]
        latencies.append(time.perf_counter() - start)
        evaluated += 1
        for rank in ranks:
            top = ranked_ids[: min(rank, len(ranked_ids))]
            if sample.identity in top:
                hits_at[rank] += 1

    cmc = {rank: (hits_at[rank] / evaluated if evaluated else 0.0) for rank in ranks}
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    throughput = (1.0 / mean_latency) if mean_latency > 0 else 0.0

    return IdentificationResult(
        rank1_accuracy=cmc.get(1, 0.0),
        cmc=cmc,
        mean_search_latency_s=mean_latency,
        throughput_probes_per_s=throughput,
        num_probes=evaluated,
        num_gallery=len(gallery),
        num_identities=len(identities),
    )


def _timed_embed(
    model: Any,
    path: Any,
    profiler: ComputeProfiler | None,
    image_transform: ImageTransform | None = None,
) -> Any:
    """Embed a path, optionally aligning then recording embedding time."""

    def _call() -> Any:
        if image_transform is None:
            return model.generate_embedding(path)
        image = load_image_rgb(path)
        return model.generate_embedding(image_transform(image))

    if profiler is None:
        return _call()
    return profiler.track_embedding(_call)
