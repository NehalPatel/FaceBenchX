"""Unit tests for environment capture and logging setup."""

from __future__ import annotations

from pathlib import Path

from facebench.utils.env_info import collect_environment_info
from facebench.utils.logging import get_logger, setup_logging


def test_collect_environment_info_has_core_fields() -> None:
    """Environment snapshot includes platform and package metadata."""
    info = collect_environment_info()
    payload = info.to_dict()
    assert "python_version" in payload
    assert "platform_system" in payload
    assert "packages" in payload
    assert "facebench" in payload["packages"]


def test_setup_logging_creates_file(tmp_path: Path) -> None:
    """Logging setup attaches a file handler when requested."""
    log_file = tmp_path / "run.log"
    logger = setup_logging(level="INFO", log_file=log_file)
    logger.info("hello-m1")
    assert log_file.is_file()
    assert "hello-m1" in log_file.read_text(encoding="utf-8")
    child = get_logger("utils.test")
    assert child.name == "facebench.utils.test"
