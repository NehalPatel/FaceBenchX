"""Optional synthetic robustness transforms on a public base dataset."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from facebench.datasets.types import IdentityPair
from facebench.evaluation.types import (
    RobustnessConditionResult,
    RobustnessReport,
    VerificationResult,
)
from facebench.evaluation.verification import run_verification
from facebench.matcher.base import BaseMatcher
from facebench.metrics import MetricCalculator
from facebench.metrics.computational import ComputeProfiler

ImageTransform = Callable[[np.ndarray], np.ndarray]

# Canonical transform names from the design / conversion brief.
SUPPORTED_TRANSFORMS: tuple[str, ...] = (
    "blur",
    "gaussian_noise",
    "jpeg",
    "low_illumination",
    "bright_illumination",
    "rotation",
    "low_resolution",
)


def get_transform(name: str) -> ImageTransform:
    """Return a named RGB transform callable.

    Args:
        name: Transform identifier.

    Returns:
        Callable mapping ``HxWx3 uint8`` → transformed array.

    Raises:
        ValueError: If the transform name is unknown.
    """
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    mapping: dict[str, ImageTransform] = {
        "blur": _blur,
        "gaussian_noise": _gaussian_noise,
        "jpeg": _jpeg,
        "low_illumination": _low_illumination,
        "bright_illumination": _bright_illumination,
        "rotation": _rotation,
        "low_resolution": _low_resolution,
        # Aliases used in design YAML examples
        "illumination": _low_illumination,
        "noise": _gaussian_noise,
        "resolution": _low_resolution,
    }
    if key not in mapping:
        supported = ", ".join(SUPPORTED_TRANSFORMS)
        raise ValueError(
            f"Unknown robustness transform {name!r}. Supported: {supported}"
        )
    return mapping[key]


def run_robustness_suite(
    pairs: list[IdentityPair],
    model: Any,
    matcher: BaseMatcher,
    *,
    transforms: Sequence[str],
    baseline: VerificationResult | None = None,
    threshold: float | None = None,
    metrics: MetricCalculator | None = None,
    base_dataset: str = "LFW",
    profiler_factory: Callable[[], ComputeProfiler] | None = None,
) -> RobustnessReport:
    """Evaluate verification under a schedule of synthetic degradations.

    Args:
        pairs: Verification pairs from a **public** base dataset.
        model: Loaded recognizer.
        matcher: Similarity matcher.
        transforms: Ordered transform names (excluding clean baseline).
        baseline: Optional precomputed clean verification result.
        threshold: Decision threshold.
        metrics: Metric calculator facade.
        base_dataset: Dataset name recorded in the report.
        profiler_factory: Factory for per-condition profilers.

    Returns:
        :class:`RobustnessReport` with deltas vs clean baseline.

    Raises:
        ValueError: If no transforms are requested.
    """
    if not transforms:
        raise ValueError("robustness.transforms must list at least one transform")

    calc = metrics or MetricCalculator()
    make_profiler = profiler_factory or (lambda: ComputeProfiler(warmup=0))

    if baseline is None:
        baseline = run_verification(
            pairs,
            model,
            matcher,
            make_profiler(),
            threshold=threshold,
            metrics=calc,
            transform_name="clean",
        )

    conditions: list[RobustnessConditionResult] = []
    for name in transforms:
        transform_fn = get_transform(name)
        result = run_verification(
            pairs,
            model,
            matcher,
            make_profiler(),
            threshold=threshold,
            metrics=calc,
            image_transform=transform_fn,
            transform_name=name,
        )
        conditions.append(
            RobustnessConditionResult(
                transform=name,
                recognition=result.recognition,
                delta_accuracy=(
                    result.recognition.accuracy - baseline.recognition.accuracy
                ),
                delta_auc=result.recognition.auc - baseline.recognition.auc,
                delta_eer=result.recognition.eer - baseline.recognition.eer,
            )
        )

    return RobustnessReport(
        baseline=baseline,
        conditions=conditions,
        base_dataset=base_dataset,
    )


def _as_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _blur(image: np.ndarray) -> np.ndarray:
    from PIL import Image, ImageFilter

    pil = Image.fromarray(image, mode="RGB")
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.uint8)


def _gaussian_noise(image: np.ndarray, *, sigma: float = 25.0) -> np.ndarray:
    noise = np.random.default_rng(0).normal(0.0, sigma, size=image.shape)
    return _as_uint8(image.astype(np.float32) + noise)


def _jpeg(image: np.ndarray, *, quality: int = 20) -> np.ndarray:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)


def _low_illumination(image: np.ndarray, *, factor: float = 0.35) -> np.ndarray:
    return _as_uint8(image.astype(np.float32) * factor)


def _bright_illumination(image: np.ndarray, *, factor: float = 1.8) -> np.ndarray:
    return _as_uint8(image.astype(np.float32) * factor)


def _rotation(image: np.ndarray, *, degrees: float = 15.0) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray(image, mode="RGB").rotate(
        degrees, expand=False, fillcolor=(0, 0, 0)
    )
    return np.asarray(pil, dtype=np.uint8)


def _low_resolution(image: np.ndarray, *, scale: float = 0.25) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray(image, mode="RGB")
    small = pil.resize(
        (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
        resample=Image.Resampling.BILINEAR,
    )
    restored = small.resize(pil.size, resample=Image.Resampling.BILINEAR)
    return np.asarray(restored, dtype=np.uint8)
