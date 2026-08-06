"""Unit tests for Buffalo-L, AdaFace, and MagFace adapters."""

from __future__ import annotations

import numpy as np
import pytest

from facebench.models import (
    AdaFaceRecognizer,
    BuffaloLRecognizer,
    DeterministicStubBackend,
    MagFaceRecognizer,
    ModelFactory,
)


def _rgb(color: tuple[int, int, int], size: int = 64) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    return image


@pytest.mark.parametrize(
    ("cls", "name", "dim"),
    [
        (BuffaloLRecognizer, "buffalo_l", 512),
        (AdaFaceRecognizer, "adaface", 512),
        (MagFaceRecognizer, "magface", 512),
    ],
)
def test_m5_adapters_with_stub_backend(cls, name: str, dim: int) -> None:
    """Each M5 adapter embeds and compares via an injected stub backend."""
    model = cls(backend=DeterministicStubBackend(embedding_dim=dim))
    model.load_model("cpu")
    assert model.name == name
    emb_a = model.generate_embedding(_rgb((200, 10, 10)))
    emb_b = model.generate_embedding(_rgb((200, 10, 10)))
    assert emb_a.shape == (dim,)
    assert model.compare(emb_a, emb_b) == pytest.approx(1.0)
    assert model.predict(emb_a, emb_b, threshold=0.5) is True


def test_m5_allow_stub_fallbacks() -> None:
    """allow_stub=True works without insightface/torch weights."""
    for cls in (BuffaloLRecognizer, AdaFaceRecognizer, MagFaceRecognizer):
        model = cls(allow_stub=True)
        model.load_model()
        embedding = model.generate_embedding(_rgb((5, 6, 7)))
        assert embedding.ndim == 1


def test_magface_last_magnitude_with_stub() -> None:
    """Stub MagFace leaves last_magnitude unset (no raw magnitude)."""
    model = MagFaceRecognizer(backend=DeterministicStubBackend(embedding_dim=512))
    model.load_model()
    model.generate_embedding(_rgb((9, 9, 9)))
    assert model.last_magnitude is None


def test_factory_creates_all_five_models() -> None:
    """ModelFactory lists and constructs all v1 recognizers."""
    factory = ModelFactory()
    assert factory.available() == [
        "adaface",
        "buffalo_l",
        "dlib",
        "facenet",
        "magface",
    ]
    for name in ("buffalo_l", "AdaFace", "MagFace"):
        model = factory.create(
            name,
            backend=DeterministicStubBackend(embedding_dim=512),
        )
        model.load_model()
        assert model.is_loaded
