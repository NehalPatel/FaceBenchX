"""Timing utilities for FaceBench profiling."""

from __future__ import annotations

import time
from types import TracebackType


class Timer:
    """Simple wall-clock timer context manager."""

    def __init__(self) -> None:
        """Initialize an inactive timer."""
        self.elapsed_seconds: float | None = None
        self._start: float | None = None

    def __enter__(self) -> Timer:
        """Start timing.

        Returns:
            The timer instance.
        """
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop timing and store elapsed seconds."""
        if self._start is not None:
            self.elapsed_seconds = time.perf_counter() - self._start
        return None
