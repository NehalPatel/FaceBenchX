"""Phase 9 benchmark execution protocol tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facebench.core.config_loader import ConfigError, ConfigLoader, load_config
from facebench.core.orchestrator import ExperimentOrchestrator
from facebench.datasets.factory import DatasetFactory
from facebench.evaluation import (
    get_transform,
    run_identification,
    run_robustness_suite,
    run_scalability_ladder,
    run_verification,
    subset_by_identities,
)
from facebench.evaluation.robustness import SUPPORTED_TRANSFORMS
from facebench.matcher import create_matcher
from facebench.metrics import MetricCalculator
from facebench.models.backends import DeterministicStubBackend
from facebench.models.factory import ModelFactory
from tests.fixtures.lfw_synthetic import make_synthetic_lfw


def _paint_jpegs(root: Path) -> None:
    for path in root.rglob("*.jpg"):
        Image.new("RGB", (48, 48), color=(30, 80, 140)).save(path, format="JPEG")


def _stub_model():
    backend = DeterministicStubBackend(embedding_dim=512)
    model = ModelFactory().create(
        "facenet", device="cpu", backend=backend, allow_stub=True
    )
    model.load_model("cpu")
    return model


def test_supported_transforms_cover_design_set() -> None:
    assert "blur" in SUPPORTED_TRANSFORMS
    assert "jpeg" in SUPPORTED_TRANSFORMS
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    for name in ("blur", "gaussian_noise", "jpeg", "low_illumination", "rotation"):
        out = get_transform(name)(image)
        assert out.shape == image.shape
        assert out.dtype == np.uint8


def test_run_verification_and_identification(tmp_path: Path) -> None:
    root = make_synthetic_lfw(tmp_path / "lfw")
    _paint_jpegs(root)
    dataset = DatasetFactory().create("LFW", root)
    pairs = dataset.load_identity_pairs()
    model = _stub_model()
    matcher = create_matcher("cosine")
    profiler = MetricCalculator().create_profiler(warmup=0)

    verification = run_verification(pairs, model, matcher, profiler, threshold=0.4)
    assert verification.num_pairs == len(pairs)
    assert 0.0 <= verification.recognition.auc <= 1.0

    identification = run_identification(
        dataset.load_gallery(),
        dataset.load_probe(),
        model,
        matcher,
    )
    assert identification.num_gallery >= 1
    assert identification.num_probes >= 1
    assert 0.0 <= identification.rank1_accuracy <= 1.0
    assert 1 in identification.cmc


def test_robustness_suite_deltas(tmp_path: Path) -> None:
    root = make_synthetic_lfw(tmp_path / "lfw")
    _paint_jpegs(root)
    pairs = DatasetFactory().create("LFW", root).load_identity_pairs()
    model = _stub_model()
    report = run_robustness_suite(
        pairs,
        model,
        create_matcher("cosine"),
        transforms=["blur", "jpeg"],
        threshold=0.4,
        base_dataset="LFW",
        profiler_factory=lambda: MetricCalculator().create_profiler(warmup=0),
    )
    assert report.baseline.transform == "clean"
    assert len(report.conditions) == 2
    assert report.conditions[0].delta_auc is not None


def test_scalability_skips_oversized_counts(tmp_path: Path) -> None:
    root = make_synthetic_lfw(tmp_path / "lfw")
    _paint_jpegs(root)
    dataset = DatasetFactory().create("LFW", root)
    ladder = run_scalability_ladder(
        dataset,
        _stub_model(),
        create_matcher("cosine"),
        identity_counts=[2, 100],
        seed=0,
    )
    assert ladder.points[0].identification is not None
    assert ladder.points[0].identification.num_identities == 2
    assert ladder.points[1].skipped_reason is not None


def test_subset_by_identities_reproducible(tmp_path: Path) -> None:
    root = make_synthetic_lfw(tmp_path / "lfw")
    _paint_jpegs(root)
    dataset = DatasetFactory().create("LFW", root)
    gallery = dataset.load_gallery()
    probe = dataset.load_probe()
    a, _, _ = subset_by_identities(gallery, probe, 2, seed=7)
    b, _, _ = subset_by_identities(gallery, probe, 2, seed=7)
    assert [s.identity for s in a] == [s.identity for s in b]


def test_config_rejects_enabled_robustness_without_transforms() -> None:
    loader = ConfigLoader()
    with pytest.raises(ConfigError, match="robustness.transforms"):
        loader.normalize(
            {
                "experiment": {"name": "bad_rob"},
                "dataset": {"name": "LFW", "root_path": "/x"},
                "model": {"name": "facenet"},
                "robustness": {"enabled": True, "transforms": []},
            }
        )


def test_example_phase9_configs_load() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("robustness_lfw.yaml", "scalability_lfw.yaml"):
        cfg = load_config(root / "configs" / "examples" / name)
        assert cfg["experiment"]["name"]


def test_orchestrator_verification_plus_axes(tmp_path: Path) -> None:
    root = make_synthetic_lfw(tmp_path / "lfw")
    _paint_jpegs(root)
    config = {
        "experiment": {
            "name": "p9_axes",
            "output_dir": str(tmp_path / "out"),
            "seed": 1,
        },
        "device": "cpu",
        "datasets": [{"name": "LFW", "root_path": str(root), "category": "general"}],
        "models": [{"name": "facenet"}],
        "matching": {"method": "cosine", "threshold": 0.4},
        "evaluation": {"mode": "verification", "max_pairs": 4},
        "robustness": {"enabled": True, "transforms": ["blur"]},
        "scalability": {"enabled": True, "identity_counts": [2, 50]},
    }
    result = ExperimentOrchestrator(
        allow_stub=True, generate_figures=False, max_pairs=4
    ).run(config)
    assert not result.dry_run
    assert len(result.reports) == 1
    extra = result.reports[0].extra
    assert "verification" in extra
    assert "robustness" in extra
    assert "scalability" in extra
    run_dir = Path(result.output_dir) / "runs"
    combo = next(run_dir.iterdir())
    assert (combo / "metrics" / "robustness.json").is_file()
    assert (combo / "metrics" / "scalability.json").is_file()


def test_orchestrator_identification_mode(tmp_path: Path) -> None:
    root = make_synthetic_lfw(tmp_path / "lfw")
    _paint_jpegs(root)
    config = {
        "experiment": {"name": "p9_id", "output_dir": str(tmp_path / "out")},
        "device": "cpu",
        "datasets": [{"name": "LFW", "root_path": str(root), "category": "general"}],
        "models": [{"name": "buffalo_l"}],
        "matching": {"method": "cosine", "threshold": 0.4},
        "evaluation": {"mode": "identification"},
        "robustness": {"enabled": False, "transforms": []},
        "scalability": {"enabled": False, "identity_counts": []},
    }
    result = ExperimentOrchestrator(allow_stub=True, generate_figures=False).run(config)
    assert result.reports[0].extra["mode"] == "identification"
    assert "identification" in result.reports[0].extra
