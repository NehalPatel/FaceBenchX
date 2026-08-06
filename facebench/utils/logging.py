"""Structured logging helpers for FaceBench experiments."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | Path | None = None,
    logger_name: str = "facebench",
) -> logging.Logger:
    """Configure and return the FaceBench logger.

    Idempotent for the named logger: existing handlers are cleared before
    new ones are attached so repeated CLI invocations do not duplicate
    log lines.

    Args:
        level: Logging level as an ``int`` or level name string.
        log_file: Optional file path for a FileHandler.
        logger_name: Root logger name for the FaceBench hierarchy.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(_coerce_level(level))
    logger.propagate = False

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``facebench`` namespace.

    Args:
        name: Optional dotted suffix (e.g. ``"core.config"``). When
            omitted, returns the root ``facebench`` logger.

    Returns:
        Logger named ``facebench`` or ``facebench.<name>``.
    """
    if name is None or name == "facebench":
        return logging.getLogger("facebench")
    if name.startswith("facebench."):
        return logging.getLogger(name)
    return logging.getLogger(f"facebench.{name}")


def _coerce_level(level: int | str) -> int:
    """Convert a level name or int into a logging level int.

    Args:
        level: Level name (e.g. ``"INFO"``) or numeric level.

    Returns:
        Numeric logging level.

    Raises:
        ValueError: If ``level`` is an unknown string.
    """
    if isinstance(level, int):
        return level
    candidate = logging.getLevelName(level.upper())
    if isinstance(candidate, int):
        return candidate
    raise ValueError(f"Unknown logging level: {level!r}")
