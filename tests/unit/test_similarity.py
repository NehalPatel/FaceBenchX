"""Unit tests for similarity helpers."""

from __future__ import annotations

import numpy as np
import pytest

from facebench.models.similarity import (
    cosine_similarity,
    euclidean_distance,
    l2_normalize,
    similarity_score,
)


def test_l2_normalize_unit_norm() -> None:
    """Normalized vectors have unit L2 norm."""
    vector = np.array([3.0, 4.0])
    normalized = l2_normalize(vector)
    assert normalized == pytest.approx(np.array([0.6, 0.8]))


def test_cosine_and_euclidean_scores() -> None:
    """Identical vectors are most similar under both methods."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, c) == pytest.approx(0.0)
    assert euclidean_distance(a, b) == pytest.approx(0.0)
    assert similarity_score(a, b, "euclidean") > similarity_score(a, c, "euclidean")
