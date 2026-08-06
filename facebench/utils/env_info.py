"""Capture runtime environment metadata for experiment reproducibility."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Any


@dataclass(slots=True)
class EnvironmentInfo:
    """Structured snapshot of the host environment.

    Attributes:
        python_version: Full Python version string.
        python_implementation: e.g. CPython.
        platform_system: OS name from :func:`platform.system`.
        platform_release: OS release string.
        platform_machine: Machine architecture.
        platform_version: Detailed platform version.
        cpu_count: Logical CPU count when available.
        hostname: Host name when available.
        cuda_available: Whether a CUDA runtime appears importable.
        cuda_version: CUDA version string if detectable.
        gpu_name: Primary GPU name if detectable.
        packages: Selected installed package versions.
        extra: Extensible bag for additional metadata.
    """

    python_version: str
    python_implementation: str
    platform_system: str
    platform_release: str
    platform_machine: str
    platform_version: str
    cpu_count: int | None
    hostname: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    packages: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this snapshot to a plain dictionary.

        Returns:
            JSON-serializable dictionary representation.
        """
        return asdict(self)


_TRACKED_PACKAGES = (
    "facebench",
    "PyYAML",
    "psutil",
    "numpy",
    "torch",
    "opencv-python",
    "insightface",
    "onnxruntime",
    "scikit-learn",
    "matplotlib",
    "seaborn",
)


def collect_environment_info(
    *,
    extra_packages: list[str] | None = None,
) -> EnvironmentInfo:
    """Collect hardware and software environment metadata.

    Args:
        extra_packages: Additional distribution names to include in the
            package version map.

    Returns:
        Populated :class:`EnvironmentInfo` instance.
    """
    cuda_available, cuda_version, gpu_name = _probe_cuda()
    packages = _collect_package_versions(extra_packages)

    return EnvironmentInfo(
        python_version=sys.version.replace("\n", " "),
        python_implementation=platform.python_implementation(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        platform_version=platform.version(),
        cpu_count=_safe_cpu_count(),
        hostname=platform.node(),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        packages=packages,
    )


def _safe_cpu_count() -> int | None:
    """Return logical CPU count, or ``None`` if unavailable.

    Returns:
        CPU count or ``None``.
    """
    try:
        import os

        return os.cpu_count()
    except Exception:  # noqa: BLE001 - best-effort environment probe
        return None


def _probe_cuda() -> tuple[bool, str | None, str | None]:
    """Best-effort CUDA / GPU detection without hard-requiring torch.

    Returns:
        Tuple of ``(cuda_available, cuda_version, gpu_name)``.
    """
    try:
        import torch
    except ImportError:
        return False, None, None

    available = bool(torch.cuda.is_available())
    version = getattr(getattr(torch, "version", None), "cuda", None)
    gpu_name: str | None = None
    if available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 - best-effort environment probe
            gpu_name = None
    return available, version, gpu_name


def _collect_package_versions(
    extra_packages: list[str] | None,
) -> dict[str, str]:
    """Resolve installed versions for tracked distributions.

    Args:
        extra_packages: Additional package names to query.

    Returns:
        Mapping of distribution name → version string (or ``"not-installed"``).
    """
    names = list(_TRACKED_PACKAGES)
    if extra_packages:
        names.extend(extra_packages)

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions
