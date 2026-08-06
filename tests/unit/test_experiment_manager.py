"""Unit tests for experiment ID creation and history tracking."""

from __future__ import annotations

from pathlib import Path

from facebench.core.experiment_manager import ExperimentManager


def test_create_experiment_writes_snapshots(tmp_path: Path) -> None:
    """Creating an experiment writes config/env snapshots and history."""
    manager = ExperimentManager(root_dir=tmp_path)
    config = {
        "experiment": {"name": "Unit Test Exp", "seed": 7, "output_dir": str(tmp_path)},
        "device": "cpu",
        "datasets": [],
        "models": [],
    }
    record = manager.create_experiment(config, environment={"python": "test"})

    exp_dir = Path(record.output_dir)
    assert exp_dir.is_dir()
    assert (exp_dir / "config.snapshot.yaml").is_file()
    assert (exp_dir / "env.json").is_file()
    assert (exp_dir / "metrics").is_dir()
    assert (exp_dir / "logs").is_dir()

    history = manager.list_history()
    assert len(history) == 1
    assert history[0].experiment_id == record.experiment_id
    assert "unit_test_exp" in record.experiment_id
