"""Dlib face recognition adapter (lightweight CPU baseline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from facebench.models.backends import DeterministicStubBackend, InferenceBackend
from facebench.models.base import BaseRecognizer
from facebench.models.imaging import ImageInput, load_image_rgb
from facebench.models.similarity import l2_normalize
from facebench.utils.logging import get_logger


class DlibBackendError(RuntimeError):
    """Raised when the Dlib backend cannot be constructed."""


class DlibFaceBackend:
    """Dlib ResNet face-recognition backend (optional dependency).

    Expects the standard dlib model files:

    * ``shape_predictor_5_face_landmarks.dat`` (or 68-point variant)
    * ``dlib_face_recognition_resnet_model_v1.dat``
    """

    embedding_dim = 128

    def __init__(
        self,
        *,
        predictor_path: str | Path | None = None,
        model_path: str | Path | None = None,
        weights_dir: str | Path | None = None,
    ) -> None:
        """Initialize the dlib backend.

        Args:
            predictor_path: Path to the shape predictor ``.dat`` file.
            model_path: Path to the recognition ResNet ``.dat`` file.
            weights_dir: Directory containing both ``.dat`` files when
                explicit paths are not provided.
        """
        self.weights_dir = (
            Path(weights_dir).expanduser() if weights_dir is not None else None
        )
        self.predictor_path = (
            Path(predictor_path).expanduser() if predictor_path is not None else None
        )
        self.model_path = (
            Path(model_path).expanduser() if model_path is not None else None
        )
        self._detector: Any = None
        self._predictor: Any = None
        self._recognizer: Any = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        """Load dlib detector, landmark predictor, and recognition model.

        Args:
            device: Accepted for API parity (dlib runs on CPU).

        Raises:
            DlibBackendError: If dlib is not installed.
            FileNotFoundError: If required ``.dat`` files are missing.
        """
        try:
            import dlib
        except ImportError as exc:
            raise DlibBackendError(
                "Dlib recognizer requires the dlib package. "
                "Install with: pip install 'facebench[dlib]'"
            ) from exc

        self.device = device
        predictor_path = self._resolve_predictor_path()
        model_path = self._resolve_model_path()
        if not predictor_path.is_file():
            raise FileNotFoundError(f"Dlib shape predictor not found: {predictor_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Dlib recognition model not found: {model_path}")

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(predictor_path))
        self._recognizer = dlib.face_recognition_model_v1(str(model_path))

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return RGB image for dlib (no tensor conversion).

        Args:
            image_rgb: ``H x W x 3`` uint8 image.

        Returns:
            Contiguous RGB array.
        """
        return np.ascontiguousarray(image_rgb)

    def embed(self, preprocessed: Any) -> np.ndarray:
        """Detect the largest face and compute a 128-D embedding.

        Args:
            preprocessed: RGB uint8 image.

        Returns:
            1-D float64 embedding.

        Raises:
            RuntimeError: If no face is detected.
        """
        if (
            self._detector is None
            or self._predictor is None
            or self._recognizer is None
        ):
            raise RuntimeError("DlibFaceBackend is not loaded")

        image = np.asarray(preprocessed)
        detections = self._detector(image, 1)
        if not detections:
            # Retry with upsampling for small faces.
            detections = self._detector(image, 2)
        if not detections:
            raise RuntimeError("DlibFaceBackend detected no faces in the image")

        face = max(detections, key=lambda rect: rect.width() * rect.height())
        shape = self._predictor(image, face)
        descriptor = self._recognizer.compute_face_descriptor(image, shape)
        vector = np.asarray(descriptor, dtype=np.float64).reshape(-1)
        return l2_normalize(vector)

    def _resolve_predictor_path(self) -> Path:
        """Resolve the landmark predictor path."""
        if self.predictor_path is not None:
            return self.predictor_path.resolve()
        if self.weights_dir is not None:
            for name in (
                "shape_predictor_5_face_landmarks.dat",
                "shape_predictor_68_face_landmarks.dat",
            ):
                candidate = self.weights_dir / name
                if candidate.is_file():
                    return candidate.resolve()
            return (self.weights_dir / "shape_predictor_5_face_landmarks.dat").resolve()
        raise FileNotFoundError(
            "Dlib predictor path not provided. Pass predictor_path= or weights_dir=."
        )

    def _resolve_model_path(self) -> Path:
        """Resolve the recognition model path."""
        if self.model_path is not None:
            return self.model_path.resolve()
        if self.weights_dir is not None:
            return (
                self.weights_dir / "dlib_face_recognition_resnet_model_v1.dat"
            ).resolve()
        raise FileNotFoundError(
            "Dlib model path not provided. Pass model_path= or weights_dir=."
        )


class DlibRecognizer(BaseRecognizer):
    """Dlib face-recognition adapter.

    Pass an explicit ``backend`` for tests, or ``allow_stub=True`` to fall
    back when dlib / weight files are unavailable.
    """

    name = "dlib"
    embedding_dim = 128

    def __init__(
        self,
        *,
        device: str = "cpu",
        weights_path: str | Path | None = None,
        predictor_path: str | Path | None = None,
        model_path: str | Path | None = None,
        backend: InferenceBackend | None = None,
        allow_stub: bool = False,
    ) -> None:
        """Initialize the Dlib recognizer.

        Args:
            device: Device string (dlib uses CPU; retained for API parity).
            weights_path: Directory containing dlib ``.dat`` model files
                (alias for ``weights_dir`` in configs).
            predictor_path: Optional explicit shape-predictor path.
            model_path: Optional explicit recognition-model path.
            backend: Optional injected inference backend.
            allow_stub: Fall back to :class:`DeterministicStubBackend` when
                dlib is unavailable.
        """
        super().__init__(device=device)
        self.weights_path = weights_path
        self.predictor_path = predictor_path
        self.model_path = model_path
        self.allow_stub = allow_stub
        self._backend = backend
        self._logger = get_logger("models.dlib")

    def load_model(self, device: str | None = None) -> None:
        """Load the Dlib backend.

        Args:
            device: Optional device override.
        """
        if device is not None:
            self.device = device
        backend = self._resolve_backend()
        try:
            backend.load(self.device)
        except (DlibBackendError, FileNotFoundError) as exc:
            if self.allow_stub and self._backend is None:
                self._logger.warning(
                    "Dlib backend unavailable (%s); using DeterministicStubBackend",
                    exc,
                )
                backend = DeterministicStubBackend(embedding_dim=128, input_size=150)
                backend.load(self.device)
            else:
                raise
        self._backend = backend
        self.embedding_dim = int(backend.embedding_dim)
        self._loaded = True
        self._logger.info(
            "Loaded Dlib backend=%s device=%s dim=%d",
            type(backend).__name__,
            self.device,
            self.embedding_dim,
        )

    def preprocess(self, image: ImageInput) -> Any:
        """Preprocess an image with the active backend.

        Args:
            image: Path or RGB ndarray.

        Returns:
            Backend-specific preprocessed input.
        """
        self.ensure_loaded()
        assert self._backend is not None
        rgb = load_image_rgb(image)
        return self._backend.preprocess(rgb)

    def generate_embedding(self, image: ImageInput) -> np.ndarray:
        """Generate a Dlib embedding.

        Args:
            image: Path or RGB ndarray.

        Returns:
            1-D L2-normalized embedding.
        """
        self.ensure_loaded()
        assert self._backend is not None
        preprocessed = self.preprocess(image)
        embedding = np.asarray(self._backend.embed(preprocessed), dtype=np.float64)
        return embedding.reshape(-1)

    def _resolve_backend(self) -> InferenceBackend:
        """Resolve the inference backend to use.

        Returns:
            Concrete backend instance.

        Raises:
            DlibBackendError: When dlib is unavailable and stubs are disallowed.
        """
        if self._backend is not None:
            return self._backend
        try:
            import dlib  # noqa: F401
        except ImportError as exc:
            if self.allow_stub:
                self._logger.warning("dlib unavailable; using DeterministicStubBackend")
                return DeterministicStubBackend(embedding_dim=128, input_size=150)
            raise DlibBackendError(
                "Dlib recognizer requires the dlib package. "
                "Install with: pip install 'facebench[dlib]' "
                "or pass allow_stub=True / an explicit backend."
            ) from exc
        return DlibFaceBackend(
            weights_dir=self.weights_path,
            predictor_path=self.predictor_path,
            model_path=self.model_path,
        )
