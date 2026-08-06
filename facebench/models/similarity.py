"""Similarity helpers shared by recognizer adapters."""

from __future__ import annotations

import numpy as np


def l2_normalize(embedding: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize an embedding vector or batch.

    Args:
        embedding: 1-D or 2-D array.
        eps: Numerical stability constant.

    Returns:
        Normalized array with the same shape.
    """
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.ndim == 1:
        norm = float(np.linalg.norm(vector))
        return vector / max(norm, eps)
    norms = np.linalg.norm(vector, axis=1, keepdims=True)
    return vector / np.maximum(norms, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        a: First embedding.
        b: Second embedding.

    Returns:
        Cosine similarity in ``[-1, 1]`` (approximately).
    """
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return float(np.dot(a_n, b_n))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two embeddings.

    Args:
        a: First embedding.
        b: Second embedding.

    Returns:
        Non-negative Euclidean distance.
    """
    a_v = np.asarray(a, dtype=np.float64).reshape(-1)
    b_v = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(a_v - b_v))


def similarity_score(
    a: np.ndarray,
    b: np.ndarray,
    method: str = "cosine",
) -> float:
    """Score two embeddings with the selected similarity method.

    Args:
        a: First embedding.
        b: Second embedding.
        method: ``cosine`` (higher is more similar) or ``euclidean``
            (returned as negative distance so higher is more similar).

    Returns:
        Similarity score where larger values mean greater similarity.

    Raises:
        ValueError: If ``method`` is unsupported.
    """
    normalized = method.strip().lower()
    if normalized == "cosine":
        return cosine_similarity(a, b)
    if normalized == "euclidean":
        return -euclidean_distance(a, b)
    raise ValueError(
        f"Unsupported similarity method {method!r}; expected 'cosine' or 'euclidean'"
    )
