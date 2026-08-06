"""Unit tests for FaceNet and Dlib recognizer adapters."""

from __future__ import annotations

import numpy as np
import pytest

from facebench.models import (
    BaseRecognizer,
    DeterministicStubBackend,
    DlibRecognizer,
    FaceNetRecognizer,
    ModelFactory,
    ModelFactoryError,
    ModelNotLoadedError,
)


def _rgb(color: tuple[int, int, int], size: int = 64) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def test_facenet_with_stub_backend_roundtrip() -> None:
    """FaceNet loads, embeds, compares, and predicts with a stub backend."""
    model = FaceNetRecognizer(backend=DeterministicStubBackend(embedding_dim=512))
    with pytest.raises(ModelNotLoadedError):
        model.generate_embedding(_rgb((10, 20, 30)))

    model.load_model("cpu")
    assert model.is_loaded
    assert isinstance(model, BaseRecognizer)

    red = model.generate_embedding(_rgb((255, 0, 0)))
    red_again = model.generate_embedding(_rgb((255, 0, 0)))
    blue = model.generate_embedding(_rgb((0, 0, 255)))

    assert red.shape == (512,)
    assert model.compare(red, red_again, method="cosine") == pytest.approx(1.0)
    assert model.compare(red, blue, method="cosine") < 0.999
    assert model.predict(red, red_again, threshold=0.5) is True


def test_dlib_with_stub_backend_roundtrip() -> None:
    """Dlib adapter works with an injected stub backend."""
    model = DlibRecognizer(backend=DeterministicStubBackend(embedding_dim=128))
    model.load_model()
    emb = model.generate_embedding(_rgb((128, 128, 128)))
    assert emb.shape == (128,)
    assert model.predict(emb, emb, threshold=0.8, method="cosine") is True


def test_facenet_allow_stub_without_torch() -> None:
    """allow_stub=True enables a deterministic fallback backend."""
    model = FaceNetRecognizer(allow_stub=True)
    model.load_model()
    embedding = model.generate_embedding(_rgb((1, 2, 3)))
    assert embedding.ndim == 1
    assert embedding.shape[0] == model.embedding_dim


def test_model_factory_creates_m4_models() -> None:
    """Factory constructs FaceNet and Dlib with stub backends."""
    factory = ModelFactory()
    assert "facenet" in factory.available()
    assert "dlib" in factory.available()

    facenet = factory.create(
        "FaceNet",
        backend=DeterministicStubBackend(embedding_dim=512),
    )
    dlib_model = factory.create(
        "dlib_fr",
        backend=DeterministicStubBackend(embedding_dim=128),
    )
    facenet.load_model()
    dlib_model.load_model()
    assert facenet.name == "facenet"
    assert dlib_model.name == "dlib"


def test_model_factory_rejects_unknown() -> None:
    """Unknown model names raise ModelFactoryError."""
    factory = ModelFactory()
    with pytest.raises(ModelFactoryError, match="Unknown model"):
        factory.create("ghostfacenet")
