"""Matcher interfaces for embedding similarity scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from facebench.models.similarity import cosine_similarity, euclidean_distance


class BaseMatcher(ABC):
    """Abstract similarity matcher."""

    name: str = "base"

    @abstractmethod
    def score(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Score two embeddings (higher means more similar).

        Args:
            embedding_a: First embedding.
            embedding_b: Second embedding.

        Returns:
            Similarity score.
        """

    def decide(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        threshold: float,
    ) -> bool:
        """Return whether the pair is predicted same-identity.

        Args:
            embedding_a: First embedding.
            embedding_b: Second embedding.
            threshold: Minimum score for a positive decision.

        Returns:
            ``True`` when ``score >= threshold``.
        """
        return self.score(embedding_a, embedding_b) >= threshold


class CosineMatcher(BaseMatcher):
    """Cosine similarity matcher."""

    name = "cosine"

    def score(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return cosine_similarity(embedding_a, embedding_b)


class EuclideanMatcher(BaseMatcher):
    """Euclidean matcher returning negative distance (higher = closer)."""

    name = "euclidean"

    def score(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute negative Euclidean distance."""
        return -euclidean_distance(embedding_a, embedding_b)


def create_matcher(method: str = "cosine") -> BaseMatcher:
    """Construct a matcher by method name.

    Args:
        method: ``cosine`` or ``euclidean``.

    Returns:
        Concrete :class:`BaseMatcher`.

    Raises:
        ValueError: If ``method`` is unsupported.
    """
    normalized = method.strip().lower()
    if normalized == "cosine":
        return CosineMatcher()
    if normalized == "euclidean":
        return EuclideanMatcher()
    raise ValueError(
        f"Unsupported matcher method {method!r}; expected 'cosine' or 'euclidean'"
    )
