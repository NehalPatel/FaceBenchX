"""Contract checks for recognizer adapters (M4 + M5)."""

from __future__ import annotations

import numpy as np
import pytest

from facebench.models.backends import DeterministicStubBackend
from facebench.models.base import BaseRecognizer
from facebench.models.factory import ModelFactory

REQUIRED_METHODS = (
    "load_model",
    "preprocess",
    "generate_embedding",
    "compare",
    "predict",
)

_MODEL_DIMS = {
    "facenet": 512,
    "dlib": 128,
    "buffalo_l": 512,
    "adaface": 512,
    "magface": 512,
}


@pytest.mark.parametrize("name", list(_MODEL_DIMS))
def test_all_recognizers_satisfy_contract(name: str) -> None:
    """Every registered recognizer exposes the BaseRecognizer API."""
    dim = _MODEL_DIMS[name]
    model = ModelFactory().create(
        name,
        backend=DeterministicStubBackend(embedding_dim=dim),
    )
    assert isinstance(model, BaseRecognizer)
    for method_name in REQUIRED_METHODS:
        assert callable(getattr(model, method_name))

    model.load_model("cpu")
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[:, :] = (40, 80, 120)
    embedding = model.generate_embedding(image)
    assert embedding.ndim == 1
    assert embedding.shape[0] == dim
    assert model.compare(embedding, embedding) == pytest.approx(1.0)
