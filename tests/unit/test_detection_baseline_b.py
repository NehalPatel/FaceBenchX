"""Unit tests for shared detection / alignment (Baseline B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facebench.core.config_loader import ConfigError, ConfigLoader, load_config
from facebench.datasets.types import IdentityPair, Sample
from facebench.detection import (
    AlignerFactoryError,
    FaceDetectionError,
    PassthroughAligner,
    as_image_transform,
    create_aligner,
)
from facebench.evaluation.verification import run_verification
from facebench.matcher import create_matcher
from facebench.metrics import MetricCalculator
from facebench.metrics.computational import ComputeProfiler
from facebench.models.backends import DeterministicStubBackend
from facebench.models.buffalo_l.recognizer import BuffaloLRecognizer


def test_create_aligner_none_is_baseline_a() -> None:
    assert create_aligner({"backend": "none"}) is None
    assert create_aligner(None) is None


def test_create_aligner_passthrough() -> None:
    aligner = create_aligner({"backend": "passthrough"})
    assert isinstance(aligner, PassthroughAligner)
    rgb = np.zeros((40, 50, 3), dtype=np.uint8)
    aligned = aligner.align(rgb)
    assert aligned.image.shape == (40, 50, 3)


def test_create_aligner_retinaface_config() -> None:
    aligner = create_aligner(
        {
            "backend": "retinaface",
            "output_size": 112,
            "det_size": [640, 640],
            "model_name": "buffalo_l",
        }
    )
    assert aligner is not None
    assert aligner.name == "retinaface"


def test_create_aligner_unknown_raises() -> None:
    with pytest.raises(AlignerFactoryError, match="Unknown detection.backend"):
        create_aligner({"backend": "yolo"})


def test_detection_config_defaults() -> None:
    loader = ConfigLoader()
    config = loader.normalize(
        {
            "experiment": {"name": "det_default"},
            "dataset": {"name": "LFW", "root_path": "/tmp/lfw"},
            "model": {"name": "buffalo_l"},
        }
    )
    assert config["detection"]["backend"] == "none"
    assert config["detection"]["output_size"] == 112
    assert config["detection"]["skip_failed"] is True


def test_detection_config_retinaface() -> None:
    loader = ConfigLoader()
    config = loader.normalize(
        {
            "experiment": {"name": "det_b"},
            "dataset": {"name": "LFW", "root_path": "/tmp/lfw"},
            "model": {"name": "buffalo_l"},
            "detection": {
                "backend": "retinaface",
                "output_size": 112,
                "det_size": [320, 320],
            },
        }
    )
    assert config["detection"]["backend"] == "retinaface"
    assert config["detection"]["det_size"] == [320, 320]


def test_detection_config_invalid_backend() -> None:
    loader = ConfigLoader()
    with pytest.raises(ConfigError, match="detection.backend"):
        loader.normalize(
            {
                "experiment": {"name": "bad"},
                "detection": {"backend": "mtcnn"},
            }
        )


def test_baseline_b_yaml_template_loads() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "paper" / "configs" / "lfw_buffalo_l_baseline_b.yaml"
    config = load_config(path)
    assert config["detection"]["backend"] == "retinaface"
    assert config["evaluation"]["max_pairs"] == 500


def test_as_image_transform_passthrough() -> None:
    transform = as_image_transform(PassthroughAligner())
    rgb = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    out = transform(rgb)
    assert out.shape == rgb.shape


def test_verification_skip_failed_detections(tmp_path: Path) -> None:
    class BoomAlign:
        name = "boom"

        def align(self, image):
            raise FaceDetectionError("no face")

    img_path = tmp_path / "face.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_path)
    sample = Sample(path=img_path, identity="id0")
    pair = IdentityPair(sample_a=sample, sample_b=sample, issame=True)

    model = BuffaloLRecognizer(backend=DeterministicStubBackend(embedding_dim=512))
    model.load_model("cpu")
    profiler = ComputeProfiler(warmup=0)
    profiler.time_model_load(lambda: None)

    transform = as_image_transform(BoomAlign())
    with pytest.raises(ValueError, match="no scored pairs"):
        run_verification(
            [pair],
            model,
            create_matcher("cosine"),
            profiler,
            metrics=MetricCalculator(),
            image_transform=transform,
            skip_failed_detections=True,
        )
