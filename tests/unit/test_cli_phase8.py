"""Phase 8 CLI and orchestrator tests."""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from facebench.core.orchestrator import ExperimentOrchestrator
from facebench.main import build_parser, main
from tests.fixtures.lfw_synthetic import make_synthetic_lfw


def _paint_jpegs(root: Path) -> None:
    """Replace placeholder files with tiny valid RGB JPEGs."""
    for path in root.rglob("*.jpg"):
        Image.new("RGB", (32, 32), color=(40, 90, 160)).save(path, format="JPEG")


def test_parser_has_phase8_commands() -> None:
    """Parser includes run/list-models/history/report."""
    parser = build_parser()
    # subparsers store choices on the command action
    sub = next(
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    )
    choices = set(sub.choices or {})
    assert {
        "run",
        "list-models",
        "history",
        "report",
        "list-datasets",
        "validate-config",
    } <= choices


def test_list_models_command() -> None:
    """list-models exits successfully."""
    assert main(["list-models"]) == 0


def test_dry_run_cli(tmp_path: Path) -> None:
    """Dry-run creates experiment metadata without evaluating pairs."""
    root = make_synthetic_lfw(tmp_path / "lfw")
    _paint_jpegs(root)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "cli_dry", "output_dir": str(tmp_path / "out")},
                "device": "cpu",
                "dataset": {"name": "LFW", "root_path": str(root)},
                "model": {"name": "facenet"},
                "matching": {"method": "cosine", "threshold": 0.4},
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "run",
                "--config",
                str(config_path),
                "--dry-run",
                "--allow-stub",
            ]
        )
        == 0
    )
    assert (
        main(["history", "--output-root", str(tmp_path / "out"), "--limit", "5"]) == 0
    )


def test_orchestrator_stub_run(tmp_path: Path) -> None:
    """Orchestrator evaluates synthetic LFW with stub backends."""
    root = make_synthetic_lfw(tmp_path / "lfw")
    _paint_jpegs(root)
    config = {
        "experiment": {"name": "orch_stub", "output_dir": str(tmp_path / "out")},
        "device": "cpu",
        "datasets": [{"name": "LFW", "root_path": str(root), "category": "general"}],
        "models": [{"name": "facenet"}],
        "matching": {"method": "cosine", "threshold": 0.4},
        "evaluation": {"aggregate_report": True, "max_pairs": 4},
    }
    result = ExperimentOrchestrator(
        allow_stub=True,
        generate_figures=False,
        max_pairs=4,
    ).run(config)
    assert result.dry_run is False
    assert len(result.reports) == 1
    assert result.reports[0].recognition is not None
    summary = Path(result.output_dir) / "metrics" / "summary.json"
    assert summary.is_file()
    assert (
        main(
            [
                "report",
                "--experiment-dir",
                result.output_dir,
                "--no-figures",
            ]
        )
        == 0
    )
