"""Abstract recognizer contract for FaceBench model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from facebench.models.imaging import ImageInput
from facebench.models.similarity import similarity_score


class ModelNotLoadedError(RuntimeError):
    """Raised when embedding APIs are used before ``load_model``."""


class BaseRecognizer(ABC):
    """Common interface implemented by every recognition model adapter.

    Concrete adapters load pretrained weights (never train) and produce
    embeddings under a shared compare/predict protocol.
    """

    #: Canonical model name (e.g. ``"facenet"``).
    name: str = "base"
    #: Default embedding dimensionality when known a priori.
    embedding_dim: int = 0

    def __init__(self, *, device: str = "cpu") -> None:
        """Initialize the recognizer.

        Args:
            device: Torch/device string such as ``cpu`` or ``cuda:0``.
        """
        self.device = device
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Return whether ``load_model`` has completed successfully."""
        return self._loaded

    @abstractmethod
    def load_model(self, device: str | None = None) -> None:
        """Load pretrained weights / runtime onto a device.

        Args:
            device: Optional device override; updates ``self.device``.
        """

    @abstractmethod
    def preprocess(self, image: ImageInput) -> Any:
        """Convert an image into the model-specific input representation.

        Args:
            image: Path or RGB ndarray.

        Returns:
            Backend-specific preprocessed object (array/tensor/batch).
        """

    @abstractmethod
    def generate_embedding(self, image: ImageInput) -> np.ndarray:
        """Generate an embedding vector for an image.

        Args:
            image: Path or RGB ndarray.

        Returns:
            1-D ``float64`` embedding vector.
        """

    def compare(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        method: str = "cosine",
    ) -> float:
        """Score two embeddings.

        Args:
            embedding_a: First embedding.
            embedding_b: Second embedding.
            method: ``cosine`` or ``euclidean`` (returned as negative
                distance so higher always means more similar).

        Returns:
            Similarity score.
        """
        return similarity_score(embedding_a, embedding_b, method=method)

    def predict(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        threshold: float,
        method: str = "cosine",
    ) -> bool:
        """Decide whether two embeddings belong to the same identity.

        Args:
            embedding_a: First embedding.
            embedding_b: Second embedding.
            threshold: Minimum similarity score for a positive decision.
            method: Similarity method forwarded to :meth:`compare`.

        Returns:
            ``True`` if ``compare(...) >= threshold``.
        """
        return self.compare(embedding_a, embedding_b, method=method) >= threshold

    def ensure_loaded(self) -> None:
        """Raise if the model has not been loaded yet.

        Raises:
            ModelNotLoadedError: When ``load_model`` has not succeeded.
        """
        if not self._loaded:
            raise ModelNotLoadedError(
                f"{self.name} model is not loaded. Call load_model() first."
            )
