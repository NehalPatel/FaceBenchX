"""InsightFace SCRFD / RetinaFace-family shared aligner (Baseline B)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from facebench.detection.align import AlignedFace, FaceDetectionError
from facebench.models.arcface_utils import resolve_device_ctx_id
from facebench.models.imaging import ImageInput, load_image_rgb
from facebench.utils.logging import get_logger


class RetinaFaceAlignerError(RuntimeError):
    """Raised when the RetinaFace/SCRFD aligner cannot be constructed."""


class RetinaFaceAligner:
    """Shared detect+align for Baseline B.

    ``crop_mode``:

    - ``bbox_margin`` (default while recognizers are unchanged): expanded
      detection box crop from the original image so vendor detectors
      (Buffalo-L, Dlib) can still re-detect.
    - ``norm_112``: ArcFace ``norm_crop`` to ``output_size`` (for future
      recognition-only adapters).
    """

    name = "retinaface"

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        output_size: int = 112,
        device: str = "cpu",
        crop_mode: str = "bbox_margin",
        bbox_margin: float = 0.35,
    ) -> None:
        """Initialize the aligner (lazy load on first :meth:`align`)."""
        self.weights_path = (
            Path(weights_path).expanduser() if weights_path is not None else None
        )
        self.model_name = model_name
        self.det_size = (int(det_size[0]), int(det_size[1]))
        self.output_size = int(output_size)
        self.device = device
        mode = str(crop_mode).strip().lower().replace("-", "_")
        if mode in {"bbox", "margin", "padded", "bbox_padded"}:
            mode = "bbox_margin"
        if mode in {"norm", "aligned", "arcface", "112"}:
            mode = "norm_112"
        if mode not in {"bbox_margin", "norm_112"}:
            raise RetinaFaceAlignerError(
                f"Unknown crop_mode {crop_mode!r}; use bbox_margin or norm_112"
            )
        self.crop_mode = mode
        self.bbox_margin = float(bbox_margin)
        self._app: Any = None
        self._face_align: Any = None
        self._logger = get_logger("detection.retinaface")

    def load(self, device: str | None = None) -> None:
        """Load InsightFace detection pack."""
        if device is not None:
            self.device = device
        try:
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align
        except ImportError as exc:
            raise RetinaFaceAlignerError(
                "RetinaFaceAligner requires insightface (and onnxruntime). "
                "Install with: pip install 'facebench[buffalo]' or "
                "pip install 'facebench[align]'"
            ) from exc

        kwargs: dict[str, Any] = {"name": self.model_name}
        if self.weights_path is not None:
            kwargs["root"] = str(self.weights_path)

        app = FaceAnalysis(**kwargs)
        app.prepare(
            ctx_id=resolve_device_ctx_id(self.device),
            det_size=self.det_size,
        )
        self._app = app
        self._face_align = face_align
        self._logger.info(
            "Loaded RetinaFaceAligner pack=%s crop_mode=%s det_size=%s "
            "output_size=%d device=%s",
            self.model_name,
            self.crop_mode,
            self.det_size,
            self.output_size,
            self.device,
        )

    def ensure_loaded(self) -> None:
        """Load the detector if it has not been loaded yet."""
        if self._app is None:
            self.load()

    def align(self, image: ImageInput) -> AlignedFace:
        """Detect the primary face and return an RGB crop."""
        self.ensure_loaded()
        assert self._app is not None
        assert self._face_align is not None

        rgb = load_image_rgb(image)
        if isinstance(image, (str, Path)):
            source = str(Path(image).expanduser().resolve())
        else:
            source = "ndarray"

        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        faces = self._app.get(bgr)
        if not faces:
            raise FaceDetectionError(
                f"RetinaFaceAligner detected no faces in image: {source}"
            )

        face = max(faces, key=lambda item: float(getattr(item, "det_score", 0.0)))
        bbox = getattr(face, "bbox", None)
        kps = getattr(face, "kps", None)

        aligned_112 = None
        if kps is not None:
            crop_bgr = self._face_align.norm_crop(
                bgr, landmark=kps, image_size=self.output_size
            )
            aligned_112 = np.ascontiguousarray(crop_bgr[:, :, ::-1])

        if self.crop_mode == "norm_112":
            if aligned_112 is None:
                raise FaceDetectionError(
                    f"RetinaFaceAligner face lacks landmarks for: {source}"
                )
            crop_rgb = aligned_112
        else:
            if bbox is None:
                raise FaceDetectionError(
                    f"RetinaFaceAligner face lacks bbox for: {source}"
                )
            crop_rgb = _bbox_margin_crop(
                rgb, np.asarray(bbox, dtype=np.float64), margin=self.bbox_margin
            )

        metadata = {
            "aligner": self.name,
            "model_name": self.model_name,
            "crop_mode": self.crop_mode,
            "output_size": self.output_size,
            "bbox_margin": self.bbox_margin,
            "det_score": float(getattr(face, "det_score", 0.0)),
            "bbox": [float(x) for x in np.asarray(bbox).reshape(-1).tolist()]
            if bbox is not None
            else None,
            "has_aligned_112": aligned_112 is not None,
        }
        return AlignedFace(image=crop_rgb, source=source, metadata=metadata)


def _bbox_margin_crop(
    rgb: np.ndarray,
    bbox: np.ndarray,
    *,
    margin: float = 0.35,
) -> np.ndarray:
    """Crop an expanded square region around a detection box."""
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox.reshape(-1)[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    half = 0.5 * side
    nx1 = int(max(0, np.floor(cx - half)))
    ny1 = int(max(0, np.floor(cy - half)))
    nx2 = int(min(width, np.ceil(cx + half)))
    ny2 = int(min(height, np.ceil(cy + half)))
    if nx2 <= nx1 or ny2 <= ny1:
        raise FaceDetectionError("Invalid bbox margin crop")
    return np.ascontiguousarray(rgb[ny1:ny2, nx1:nx2])
