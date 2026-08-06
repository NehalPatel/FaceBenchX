"""AdaFace recognizer adapter (quality-adaptive ArcFace variant)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from facebench.models.arcface_utils import embedding_from_torch, preprocess_arcface_112
from facebench.models.backends import DeterministicStubBackend, InferenceBackend
from facebench.models.base import BaseRecognizer
from facebench.models.imaging import ImageInput, load_image_rgb
from facebench.utils.logging import get_logger


class AdaFaceBackendError(RuntimeError):
    """Raised when the AdaFace backend cannot be constructed."""


class TorchAdaFaceBackend:
    """Torch checkpoint backend for AdaFace.

    Expects a loadable ``torch.nn.Module`` checkpoint (``state_dict`` or
    full model) at ``weights_path``. When only a ``state_dict`` is provided,
    pass a constructed ``model`` module via the constructor.
    """

    embedding_dim = 512

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        """Initialize the AdaFace torch backend.

        Args:
            weights_path: Path to a ``.pt`` / ``.pth`` checkpoint.
            model: Optional pre-built ``nn.Module`` that accepts a
                ``1x3x112x112`` tensor and returns embeddings.
        """
        self.weights_path = (
            Path(weights_path).expanduser() if weights_path is not None else None
        )
        self._model = model
        self.device = "cpu"

    def load(self, device: str) -> None:
        """Load AdaFace weights onto ``device``.

        Args:
            device: Torch device string.

        Raises:
            AdaFaceBackendError: If torch is unavailable or no model/weights
                can be resolved.
            FileNotFoundError: If ``weights_path`` does not exist.
        """
        try:
            import torch
        except ImportError as exc:
            raise AdaFaceBackendError(
                "AdaFace requires torch. Install with: pip install 'facebench[adaface]'"
            ) from exc

        self.device = device
        if self._model is None:
            if self.weights_path is None:
                raise AdaFaceBackendError(
                    "AdaFace requires weights_path= to a pretrained checkpoint "
                    "or an injected model= module. Official AdaFace weights are "
                    "not bundled with FaceBench."
                )
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    f"AdaFace weights not found: {self.weights_path}"
                )
            loaded = torch.load(str(self.weights_path), map_location=device)
            if hasattr(loaded, "eval") and callable(loaded.eval):
                self._model = loaded
            else:
                raise AdaFaceBackendError(
                    "AdaFace checkpoint does not contain a full nn.Module. "
                    "Construct the architecture externally and pass model=, "
                    "or provide a full serialized module checkpoint."
                )
        elif self.weights_path is not None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    f"AdaFace weights not found: {self.weights_path}"
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
        """Run AdaFace forward pass."""
        import torch

        if self._model is None:
            raise RuntimeError("TorchAdaFaceBackend is not loaded")
        with torch.no_grad():
            output = self._model(preprocessed)
            if isinstance(output, (tuple, list)):
                output = output[0]
        vector = embedding_from_torch(output)
        self.embedding_dim = int(vector.size)
        return vector


class AdaFaceRecognizer(BaseRecognizer):
    """AdaFace recognition adapter.

    Production use requires a local AdaFace checkpoint (``weights_path``)
    and optional architecture module. Tests may inject
    :class:`DeterministicStubBackend` or set ``allow_stub=True``.
    """

    name = "adaface"
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
        """Initialize AdaFace.

        Args:
            device: Target device string.
            weights_path: Local AdaFace checkpoint path.
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
        self._logger = get_logger("models.adaface")

    def load_model(self, device: str | None = None) -> None:
        """Load the AdaFace backend."""
        if device is not None:
            self.device = device
        backend = self._resolve_backend()
        try:
            backend.load(self.device)
        except (AdaFaceBackendError, FileNotFoundError, ImportError) as exc:
            if self.allow_stub and self._backend is None:
                self._logger.warning(
                    "AdaFace backend unavailable (%s); using stub", exc
                )
                backend = DeterministicStubBackend(embedding_dim=512, input_size=112)
                backend.load(self.device)
            else:
                raise
        self._backend = backend
        self.embedding_dim = int(backend.embedding_dim)
        self._loaded = True
        self._logger.info(
            "Loaded AdaFace backend=%s device=%s dim=%d",
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
        """Generate an AdaFace embedding."""
        self.ensure_loaded()
        assert self._backend is not None
        preprocessed = self.preprocess(image)
        return np.asarray(self._backend.embed(preprocessed), dtype=np.float64).reshape(
            -1
        )

    def _resolve_backend(self) -> InferenceBackend:
        """Resolve torch AdaFace or stub backend."""
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
            raise AdaFaceBackendError(
                "AdaFace requires torch. "
                "Install with: pip install 'facebench[adaface]' "
                "or pass allow_stub=True / an explicit backend."
            ) from exc
        return TorchAdaFaceBackend(weights_path=self.weights_path, model=self.model)
