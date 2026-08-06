"""Recognition model adapters.

Milestone M5 provides FaceNet, Dlib, Buffalo-L, AdaFace, and MagFace
adapters via :class:`BaseRecognizer` and :class:`ModelFactory`.
Metrics and reports arrive in later milestones.
"""

from __future__ import annotations

from facebench.models.adaface import AdaFaceBackendError, AdaFaceRecognizer
from facebench.models.backends import DeterministicStubBackend
from facebench.models.base import BaseRecognizer, ModelNotLoadedError
from facebench.models.buffalo_l import BuffaloLBackendError, BuffaloLRecognizer
from facebench.models.dlib_fr import DlibBackendError, DlibRecognizer
from facebench.models.facenet import FaceNetBackendError, FaceNetRecognizer
from facebench.models.factory import ModelFactory, ModelFactoryError, create_model
from facebench.models.magface import MagFaceBackendError, MagFaceRecognizer
from facebench.models.similarity import (
    cosine_similarity,
    euclidean_distance,
    l2_normalize,
    similarity_score,
)

__all__ = [
    "AdaFaceBackendError",
    "AdaFaceRecognizer",
    "BaseRecognizer",
    "BuffaloLBackendError",
    "BuffaloLRecognizer",
    "DeterministicStubBackend",
    "DlibBackendError",
    "DlibRecognizer",
    "FaceNetBackendError",
    "FaceNetRecognizer",
    "MagFaceBackendError",
    "MagFaceRecognizer",
    "ModelFactory",
    "ModelFactoryError",
    "ModelNotLoadedError",
    "cosine_similarity",
    "create_model",
    "euclidean_distance",
    "l2_normalize",
    "similarity_score",
]
