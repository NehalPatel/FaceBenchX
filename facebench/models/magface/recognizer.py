"""MagFace recognizer adapter (magnitude-aware representation learning)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from facebench.models.arcface_utils import embedding_from_torch, preprocess_arcface_112
from facebench.models.backends import DeterministicStubBackend, InferenceBackend
from facebench.models.base import BaseRecognizer
from facebench.models.imaging import ImageInput, load_image_rgb
from facebench.utils.logging import get_logger


class MagFaceBackendError(RuntimeError):
    """Raised when the MagFace backend cannot be constructed."""


class TorchMagFaceBackend:
    """Torch checkpoint backend for MagFace.

    MagFace embeddings encode quality in vector magnitude. This backend
    exposes the raw (pre-normalization) magnitude via ``last_magnitude``
    while still returning an L2-normalized embedding for matching.
    """

    embedding_dim = 512

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        """Initialize the MagFace torch backend.

        Args:
            weights_path: Path to a MagFace ``.pt`` / ``.pth`` checkpoint.
            model: Optional pre-built ``nn.Module``.
        """
        self.weights_path = (
            Path(weights_path).expanduser() if weights_path is not None else None
        )
        self._model = model
        self.device = "cpu"
        self.last_magnitude: float | None = None

    def load(self, device: str) -> None:
        """Load MagFace weights onto ``device``.

        Args:
            device: Torch device string.

        Raises:
            MagFaceBackendError: If torch is unavailable or no model/weights
                can be resolved.
            FileNotFoundError: If ``weights_path`` does not exist.
        """
        try:
            import torch
        except ImportError as exc:
            raise MagFaceBackendError(
                "MagFace requires torch. Install with: pip install 'facebench[magface]'"
            ) from exc

        self.device = device
        if self._model is None:
            if self.weights_path is None:
                raise MagFaceBackendError(
                    "MagFace requires weights_path= to a pretrained checkpoint "
                    "or an injected model= module. Official MagFace weights are "
                    "not bundled with FaceBench."
                )
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    f"MagFace weights not found: {self.weights_path}"
                )
            loaded = torch.load(str(self.weights_path), map_location=device)
            if hasattr(loaded, "eval") and callable(loaded.eval):
                self._model = loaded
            else:
                raise MagFaceBackendError(
                    "MagFace checkpoint does not contain a full nn.Module. "
                    "Construct the architecture externally and pass model=, "
                    "or provide a full serialized module checkpoint."
                )
        elif self.weights_path is not None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    f"MagFace weights not found: {self.weights_path}"
                )
            state = torch.load(str(self.weights_path), map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self._model.load_state_dict(state, strict=False)

        self._model.to(device)
        self._model.eval()

    def preprocess(self, image_rgb: np.ndarray) -> Any:
        """Preprocess RGB image to ArcFace tensor batch."""
        return preprocess_arcface_112(image_rgb, device=self.device)

    def embed(self, preprocessed: Any) -> np.ndarray:
        """Run MagFace forward pass and record embedding magnitude."""
        import torch

        if self._model is None:
            raise RuntimeError("TorchMagFaceBackend is not loaded")
        with torch.no_grad():
            output = self._model(preprocessed)
            if isinstance(output, (tuple, list)):
                output = output[0]
        raw = output.detach().cpu().numpy().reshape(-1).astype(np.float64)
        self.last_magnitude = float(np.linalg.norm(raw))
        vector = embedding_from_torch(output)
        self.embedding_dim = int(vector.size)
        return vector


class MagFaceRecognizer(BaseRecognizer):
    """MagFace recognition adapter.

    After :meth:`generate_embedding`, :attr:`last_magnitude` exposes the
    pre-normalization L2 norm when the torch MagFace backend is used
    (useful as a quality signal).
    """

    name = "magface"
    embedding_dim = 512

    def __init__(
        self,
        *,
        device: str = "cpu",
        weights_path: str | Path | None = None,
        model: Any | None = None,
        backend: InferenceBackend | None = None,
        allow_stub: bool = False,
    ) -> None:
        """Initialize MagFace.

        Args:
            device: Target device string.
            weights_path: Local MagFace checkpoint path.
            model: Optional pre-built torch module.
            backend: Optional injected backend.
            allow_stub: Fall back to deterministic stub when deps/weights
                are unavailable.
        """
        super().__init__(device=device)
        self.weights_path = weights_path
        self.model = model
        self.allow_stub = allow_stub
        self._backend = backend
        self.last_magnitude: float | None = None
        self._logger = get_logger("models.magface")

    def load_model(self, device: str | None = None) -> None:
        """Load the MagFace backend."""
        if device is not None:
            self.device = device
        backend = self._resolve_backend()
        try:
            backend.load(self.device)
        except (MagFaceBackendError, FileNotFoundError, ImportError) as exc:
            if self.allow_stub and self._backend is None:
                self._logger.warning(
                    "MagFace backend unavailable (%s); using stub", exc
                )
                backend = DeterministicStubBackend(embedding_dim=512, input_size=112)
                backend.load(self.device)
            else:
                raise
        self._backend = backend
        self.embedding_dim = int(backend.embedding_dim)
        self._loaded = True
        self._logger.info(
            "Loaded MagFace backend=%s device=%s dim=%d",
            type(backend).__name__,
            self.device,
            self.embedding_dim,
        )

    def preprocess(self, image: ImageInput) -> Any:
        """Preprocess an image with the active backend."""
        self.ensure_loaded()
        assert self._backend is not None
        return self._backend.preprocess(load_image_rgb(image))

    def generate_embedding(self, image: ImageInput) -> np.ndarray:
        """Generate a MagFace embedding and update ``last_magnitude``."""
        self.ensure_loaded()
        assert self._backend is not None
        preprocessed = self.preprocess(image)
        embedding = np.asarray(
            self._backend.embed(preprocessed), dtype=np.float64
        ).reshape(-1)
        magnitude = getattr(self._backend, "last_magnitude", None)
        self.last_magnitude = float(magnitude) if magnitude is not None else None
        return embedding

    def _resolve_backend(self) -> InferenceBackend:
        """Resolve torch MagFace or stub backend."""
        if self._backend is not None:
            return self._backend
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            if self.allow_stub:
                self._logger.warning(
                    "torch unavailable; using DeterministicStubBackend"
                )
                return DeterministicStubBackend(embedding_dim=512, input_size=112)
            raise MagFaceBackendError(
                "MagFace requires torch. "
                "Install with: pip install 'facebench[magface]' "
                "or pass allow_stub=True / an explicit backend."
            ) from exc
        return TorchMagFaceBackend(weights_path=self.weights_path, model=self.model)
