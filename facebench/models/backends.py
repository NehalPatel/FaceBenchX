"""Pluggable inference backends for recognizer adapters."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class InferenceBackend(Protocol):
    """Protocol implemented by concrete model backends."""

    embedding_dim: int

    def load(self, device: str) -> None:
        """Load weights/runtime onto ``device``."""

    def preprocess(self, image_rgb: np.ndarray) -> Any:
        """Preprocess an RGB ``uint8`` image."""

    def embed(self, preprocessed: Any) -> np.ndarray:
        """Generate a 1-D embedding from preprocessed input."""


class DeterministicStubBackend:
    """Lightweight deterministic backend used for unit tests and smoke runs.

    Embeddings are derived from resized image statistics so identical images
    score highly and dissimilar random images score lower — without requiring
    torch, dlib, or pretrained weights.
    """

    def __init__(self, *, embedding_dim: int = 128, input_size: int = 160) -> None:
        """Initialize the stub backend.

        Args:
            embedding_dim: Output embedding size.
            input_size: Square resize side length used before featurization.
        """
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self._loaded = False
        self.device = "cpu"

    def load(self, device: str) -> None:
        """Mark the stub backend as loaded.

        Args:
            device: Device string stored for API parity.
        """
        self.device = device
        self._loaded = True

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Resize and scale an RGB image to ``float32`` ``[0, 1]``.

        Args:
            image_rgb: ``H x W x 3`` uint8 image.

        Returns:
            ``input_size x input_size x 3`` float32 array.
        """
        try:
            from PIL import Image
        except ImportError:
            # Nearest-neighbor fallback without Pillow.
            return self._resize_numpy(image_rgb)

        image = Image.fromarray(image_rgb)
        resized = image.resize((self.input_size, self.input_size))
        return np.asarray(resized, dtype=np.float32) / 255.0

    def embed(self, preprocessed: Any) -> np.ndarray:
        """Create a deterministic embedding from preprocessed pixels.

        Args:
            preprocessed: Float image array from :meth:`preprocess`.

        Returns:
            L2-normalized 1-D embedding.
        """
        if not self._loaded:
            raise RuntimeError("DeterministicStubBackend is not loaded")
        array = np.asarray(preprocessed, dtype=np.float64)
        flat = array.reshape(-1)
        # Fixed pseudo-random projection for stable test vectors.
        rng = np.random.default_rng(0)
        basis = rng.standard_normal((flat.size, self.embedding_dim))
        vector = flat @ basis
        # Mix in channel means for additional image sensitivity.
        channel_means = array.mean(axis=(0, 1))
        vector[:3] += channel_means
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def _resize_numpy(self, image_rgb: np.ndarray) -> np.ndarray:
        """Simple nearest-neighbor resize without Pillow."""
        height, width = image_rgb.shape[:2]
        y_idx = (np.linspace(0, height - 1, self.input_size)).astype(int)
        x_idx = (np.linspace(0, width - 1, self.input_size)).astype(int)
        resized = image_rgb[y_idx][:, x_idx]
        return resized.astype(np.float32) / 255.0
