"""Unit tests for YAML configuration loading and normalization."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from facebench.core.config_loader import ConfigError, ConfigLoader, load_config


def test_load_smoke_config(tmp_path: Path) -> None:
    """Round-trip: write YAML, load, and assert normalized fields."""
    path = tmp_path / "exp.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "roundtrip", "output_dir": "experiments"},
                "device": "cpu",
                "dataset": {"name": "LFW", "root_path": "/tmp/lfw"},
                "model": {"name": "buffalo_l"},
                "matching": {"method": "cosine", "threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config["experiment"]["name"] == "roundtrip"
    assert config["device"] == "cpu"
    assert len(config["datasets"]) == 1
    assert config["datasets"][0]["name"] == "LFW"
    assert config["datasets"][0]["category"] == "general"
    assert config["models"][0]["name"] == "buffalo_l"
    assert config["matching"]["method"] == "cosine"
    assert config["_meta"]["source_path"] == str(path.resolve())


def test_datasets_all_expansion() -> None:
    """``datasets: all`` expands via the category registry."""
    loader = ConfigLoader()
    config = loader.normalize(
        {
            "experiment": {"name": "all_ds"},
            "datasets": "all",
            "dataset_roots": {"LFW": "/data/lfw"},
            "model": {"name": "dlib"},
        }
    )
    names = [entry["name"] for entry in config["datasets"]]
    assert "LFW" in names
    assert "YTF" in names
    assert len(names) == 8


def test_missing_experiment_name_raises() -> None:
    """Missing experiment.name must raise ConfigError."""
    loader = ConfigLoader()
    with pytest.raises(ConfigError, match="experiment.name"):
        loader.normalize({"experiment": {}})


def test_unsupported_dataset_raises() -> None:
    """Custom / unknown datasets are rejected."""
    loader = ConfigLoader()
    with pytest.raises(ConfigError, match="Unsupported dataset"):
        loader.normalize(
            {
                "experiment": {"name": "bad"},
                "dataset": {"name": "MyPrivateSet", "root_path": "/x"},
            }
        )


def test_example_smoke_config_loads() -> None:
    """Repository example config must load successfully."""
    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "examples" / "smoke_m1.yaml"
    config = load_config(path)
    assert config["experiment"]["name"] == "smoke_m1"
