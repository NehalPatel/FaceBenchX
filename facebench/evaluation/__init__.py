"""Verification, identification, robustness, and scalability protocols."""

from __future__ import annotations

from facebench.evaluation.identification import run_identification
from facebench.evaluation.robustness import (
    SUPPORTED_TRANSFORMS,
    get_transform,
    run_robustness_suite,
)
from facebench.evaluation.scalability import (
    DEFAULT_IDENTITY_COUNTS,
    run_scalability_ladder,
    subset_by_identities,
)
from facebench.evaluation.types import (
    IdentificationResult,
    RobustnessConditionResult,
    RobustnessReport,
    ScalabilityPoint,
    ScalabilityReport,
    VerificationResult,
)
from facebench.evaluation.verification import run_verification

__all__ = [
    "DEFAULT_IDENTITY_COUNTS",
    "SUPPORTED_TRANSFORMS",
    "IdentificationResult",
    "RobustnessConditionResult",
    "RobustnessReport",
    "ScalabilityPoint",
    "ScalabilityReport",
    "VerificationResult",
    "get_transform",
    "run_identification",
    "run_robustness_suite",
    "run_scalability_ladder",
    "run_verification",
    "subset_by_identities",
]
