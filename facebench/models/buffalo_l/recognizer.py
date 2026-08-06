"""Buffalo-L (InsightFace) recognizer adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from facebench.models.arcface_utils import resolve_device_ctx_id
from facebench.models.backends import DeterministicStubBackend, InferenceBackend
from facebench.models.base import BaseRecognizer
from facebench.models.imaging import ImageInput, load_image_rgb
from facebench.models.similarity import l2_normalize
from facebench.utils.logging import get_logger


class BuffaloLBackendError(RuntimeError):
    """Raised when the Buffalo-L backend cannot be constructed."""


class InsightFaceBuffaloBackend:
    """Buffalo-L backend via InsightFace ``FaceAnalysis``."""

    embedding_dim = 512

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        """Initialize the InsightFace Buffalo-L backend.

        Args:
            weights_path: Optional InsightFace models root directory
                (passed as ``root`` to ``FaceAnalysis``).
            model_name: InsightFace pack name (default ``buffalo_l``).
            det_size: Detector input size.
        """
        self.weights_path = (
            Path(weights_path).expanduser() if weights_path is not None else None
        )
        self.model_name = model_name
        self.det_size = det_size
        self._app: Any = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        """Load Buffalo-L detection + recognition pack.

        Args:
            device: FaceBench device string.

        Raises:
            BuffaloLBackendError: If insightface / onnxruntime are missing.
        """
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise BuffaloLBackendError(
                "Buffalo-L requires insightface (and onnxruntime). "
                "Install with: pip install 'facebench[buffalo]'"
            ) from exc

        self.device = device
        kwargs: dict[str, Any] = {"name": self.model_name}
        if self.weights_path is not None:
            kwargs["root"] = str(self.weights_path)
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=resolve_device_ctx_id(device), det_size=self.det_size)
        self._app = app

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to BGR contiguous array expected by InsightFace.

        Args:
            image_rgb: ``H x W x 3`` uint8 image.

        Returns:
            BGR ``uint8`` array.
        """
        return np.ascontiguousarray(image_rgb[:, :, ::-1])

    def embed(self, preprocessed: Any) -> np.ndarray:
        """Detect the primary face and return its embedding.

        Args:
            preprocessed: BGR image from :meth:`preprocess`.

        Returns:
            1-D L2-normalized embedding.

        Raises:
            RuntimeError: If no face is detected.
        """
        if self._app is None:
            raise RuntimeError("InsightFaceBuffaloBackend is not loaded")
        faces = self._app.get(preprocessed)
        if not faces:
            raise RuntimeError("Buffalo-L detected no faces in the image")
        face = max(faces, key=lambda item: float(getattr(item, "det_score", 0.0)))
        vector = np.asarray(face.embedding, dtype=np.float64).reshape(-1)
        self.embedding_dim = int(vector.size)
        return l2_normalize(vector)


class BuffaloLRecognizer(BaseRecognizer):
    """Buffalo-L recognition adapter (InsightFace / ArcFace production pack).

    Pass an explicit ``backend`` for tests, or ``allow_stub=True`` when
    InsightFace is unavailable.
    """

    name = "buffalo_l"
    embedding_dim = 512

    def __init__(
        self,
        *,
        device: str = "cpu",
        weights_path: str | Path | None = None,
        model_name: str = "buffalo_l",
        backend: InferenceBackend | None = None,
        allow_stub: bool = False,
    ) -> None:
        """Initialize Buffalo-L.

        Args:
            device: Target device string.
            weights_path: Optional InsightFace models root.
            model_name: InsightFace pack name.
            backend: Optional injected backend.
            allow_stub: Fall back to :class:`DeterministicStubBackend`.
        """
        super().__init__(device=device)
        self.weights_path = weights_path
        self.model_name = model_name
        self.allow_stub = allow_stub
        self._backend = backend
        self._logger = get_logger("models.buffalo_l")

    def load_model(self, device: str | None = None) -> None:
        """Load the Buffalo-L backend.

        Args:
            device: Optional device override.
        """
        if device is not None:
            self.device = device
        backend = self._resolve_backend()
        try:
            backend.load(self.device)
        except BuffaloLBackendError:
            if self.allow_stub and not isinstance(backend, DeterministicStubBackend):
                self._logger.warning(
                    "InsightFace unavailable; using DeterministicStubBackend"
                )
                backend = DeterministicStubBackend(embedding_dim=512, input_size=112)
                backend.load(self.device)
            else:
                raise
        self._backend = backend
        self.embedding_dim = int(backend.embedding_dim)
        self._loaded = True
        self._logger.info(
            "Loaded Buffalo-L backend=%s device=%s dim=%d",
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
        """Generate a Buffalo-L embedding."""
        self.ensure_loaded()
        assert self._backend is not None
        preprocessed = self.preprocess(image)
        return np.asarray(self._backend.embed(preprocessed), dtype=np.float64).reshape(
            -1
        )

    def _resolve_backend(self) -> InferenceBackend:
        """Resolve InsightFace or stub backend."""
        if self._backend is not None:
            return self._backend
        try:
            import insightface  # noqa: F401
        except ImportError as exc:
            if self.allow_stub:
                self._logger.warning(
                    "insightface unavailable; using DeterministicStubBackend"
                )
                return DeterministicStubBackend(embedding_dim=512, input_size=112)
            raise BuffaloLBackendError(
                "Buffalo-L requires insightface. "
                "Install with: pip install 'facebench[buffalo]' "
                "or pass allow_stub=True / an explicit backend."
            ) from exc
        return InsightFaceBuffaloBackend(
            weights_path=self.weights_path,
            model_name=self.model_name,
        )
