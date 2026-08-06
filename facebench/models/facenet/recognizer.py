"""FaceNet recognizer adapter (historical triplet-loss baseline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from facebench.models.backends import DeterministicStubBackend, InferenceBackend
from facebench.models.base import BaseRecognizer
from facebench.models.imaging import ImageInput, load_image_rgb
from facebench.models.similarity import l2_normalize
from facebench.utils.logging import get_logger


class FaceNetBackendError(RuntimeError):
    """Raised when the FaceNet backend cannot be constructed."""


class TorchFaceNetBackend:
    """FaceNet backend backed by ``facenet-pytorch`` (optional dependency)."""

    embedding_dim = 512

    def __init__(self, weights_path: str | Path | None = None) -> None:
        """Initialize the torch FaceNet backend.

        Args:
            weights_path: Optional local checkpoint path. When omitted,
                ``facenet-pytorch`` pretrained weights are requested.
        """
        self.weights_path = (
            Path(weights_path).expanduser() if weights_path is not None else None
        )
        self._model: Any = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        """Load InceptionResnetV1 onto ``device``.

        Args:
            device: Torch device string.

        Raises:
            FaceNetBackendError: If torch / facenet-pytorch are unavailable.
            FileNotFoundError: If ``weights_path`` is set but missing.
        """
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:
            raise FaceNetBackendError(
                "FaceNet requires torch and facenet-pytorch. "
                "Install with: pip install 'facebench[facenet]'"
            ) from exc

        self.device = device
        pretrained = "vggface2" if self.weights_path is None else None
        model = InceptionResnetV1(pretrained=pretrained).eval()
        if self.weights_path is not None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    f"FaceNet weights not found: {self.weights_path}"
                )
            state = torch.load(str(self.weights_path), map_location=device)
            model.load_state_dict(state)
        model.to(device)
        self._model = model

    def preprocess(self, image_rgb: np.ndarray) -> Any:
        """Convert RGB uint8 image to a normalized torch tensor batch.

        Args:
            image_rgb: ``H x W x 3`` image.

        Returns:
            Tensor shaped ``1 x 3 x 160 x 160``.
        """
        from PIL import Image
        from torchvision import transforms

        image = Image.fromarray(image_rgb)
        transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        tensor = transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def embed(self, preprocessed: Any) -> np.ndarray:
        """Run FaceNet forward pass.

        Args:
            preprocessed: Tensor batch from :meth:`preprocess`.

        Returns:
            1-D float64 embedding.
        """
        import torch

        if self._model is None:
            raise RuntimeError("TorchFaceNetBackend is not loaded")
        with torch.no_grad():
            embedding = self._model(preprocessed)
        vector = embedding.detach().cpu().numpy().reshape(-1).astype(np.float64)
        return l2_normalize(vector)


class FaceNetRecognizer(BaseRecognizer):
    """FaceNet recognition adapter.

    By default attempts to use ``facenet-pytorch``. Pass an explicit
    ``backend`` (e.g. :class:`DeterministicStubBackend`) for tests, or set
    ``allow_stub=True`` to fall back when optional deps are missing.
    """

    name = "facenet"
    embedding_dim = 512

    def __init__(
        self,
        *,
        device: str = "cpu",
        weights_path: str | Path | None = None,
        backend: InferenceBackend | None = None,
        allow_stub: bool = False,
    ) -> None:
        """Initialize FaceNet.

        Args:
            device: Target device string.
            weights_path: Optional local weight file.
            backend: Optional injected inference backend.
            allow_stub: When ``True`` and no backend is provided, fall back
                to :class:`DeterministicStubBackend` if torch FaceNet cannot
                be imported.
        """
        super().__init__(device=device)
        self.weights_path = weights_path
        self.allow_stub = allow_stub
        self._backend = backend
        self._logger = get_logger("models.facenet")

    def load_model(self, device: str | None = None) -> None:
        """Load the FaceNet backend.

        Args:
            device: Optional device override.
        """
        if device is not None:
            self.device = device
        backend = self._resolve_backend()
        backend.load(self.device)
        self._backend = backend
        self.embedding_dim = int(backend.embedding_dim)
        self._loaded = True
        self._logger.info(
            "Loaded FaceNet backend=%s device=%s dim=%d",
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
        """Generate a FaceNet embedding.

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
            FaceNetBackendError: When no usable backend is available.
        """
        if self._backend is not None:
            return self._backend
        try:
            import facenet_pytorch  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            if self.allow_stub:
                self._logger.warning(
                    "facenet-pytorch unavailable; using DeterministicStubBackend"
                )
                return DeterministicStubBackend(embedding_dim=512, input_size=160)
            raise FaceNetBackendError(
                "FaceNet requires torch and facenet-pytorch. "
                "Install with: pip install 'facebench[facenet]' "
                "or pass allow_stub=True / an explicit backend."
            ) from exc
        return TorchFaceNetBackend(weights_path=self.weights_path)
