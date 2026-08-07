"""Verification (1:1) benchmark protocol."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from facebench.datasets.types import IdentityPair
from facebench.detection.align import FaceDetectionError
from facebench.evaluation.types import VerificationResult
from facebench.matcher.base import BaseMatcher
from facebench.metrics import MetricCalculator
from facebench.metrics.computational import ComputeProfiler
from facebench.models.imaging import load_image_rgb

ImageTransform = Callable[[np.ndarray], np.ndarray]


def run_verification(
    pairs: list[IdentityPair],
    model: Any,
    matcher: BaseMatcher,
    profiler: ComputeProfiler,
    *,
    threshold: float | None = None,
    metrics: MetricCalculator | None = None,
    image_transform: ImageTransform | None = None,
    transform_name: str | None = None,
    skip_failed_detections: bool = False,
) -> VerificationResult:
    """Score identity pairs and compute recognition + compute metrics.

    Args:
        pairs: Same/different verification pairs.
        model: Recognizer exposing ``generate_embedding``.
        matcher: Similarity matcher.
        profiler: Active compute profiler (model load already timed).
        threshold: Optional fixed decision threshold.
        metrics: Metric calculator facade; created if omitted.
        image_transform: Optional RGB ndarray transform applied before embed.
        transform_name: Label stored on the result (e.g. ``blur``).
        skip_failed_detections: When ``True``, skip pairs where embedding
            fails due to missed face detection instead of aborting.

    Returns:
        :class:`VerificationResult`.

    Raises:
        ValueError: If ``pairs`` is empty, or if every pair is skipped.
        FaceDetectionError / RuntimeError: On detection failure when
            ``skip_failed_detections`` is ``False``.
    """
    if not pairs:
        raise ValueError("Verification requires at least one identity pair")

    calc = metrics or MetricCalculator()
    labels: list[int] = []
    scores: list[float] = []
    skipped = 0

    for pair in pairs:
        try:
            emb_a = profiler.track_embedding(
                lambda p=pair.sample_a.path: _embed(model, p, image_transform)
            )
            emb_b = profiler.track_embedding(
                lambda p=pair.sample_b.path: _embed(model, p, image_transform)
            )
        except Exception as exc:
            if skip_failed_detections and _is_detection_failure(exc):
                skipped += 1
                continue
            raise
        scores.append(float(matcher.score(emb_a, emb_b)))
        labels.append(1 if pair.issame else 0)

    if not labels:
        raise ValueError(
            "Verification produced no scored pairs "
            f"(skipped_failed_detections={skipped})"
        )

    recognition = calc.recognition(
        np.asarray(labels),
        np.asarray(scores, dtype=np.float64),
        threshold=threshold,
    )
    result = VerificationResult(
        recognition=recognition,
        computational=profiler.summarize(),
        labels=labels,
        scores=scores,
        num_pairs=len(labels),
        transform=transform_name,
    )
    if skipped:
        result.extra["skipped_failed_detections"] = skipped
        result.extra["pairs_requested"] = len(pairs)
    return result


def _is_detection_failure(exc: BaseException) -> bool:
    """Return True for shared/vendor missed-face errors."""
    if isinstance(exc, FaceDetectionError):
        return True
    message = str(exc).lower()
    return "no face" in message or "detected no faces" in message


def _embed(
    model: Any,
    path: Path,
    image_transform: ImageTransform | None,
) -> Any:
    """Generate an embedding, optionally transforming the loaded image."""
    if image_transform is None:
        return model.generate_embedding(path)
    image = load_image_rgb(path)
    return model.generate_embedding(image_transform(image))
