"""Experiment identity, directory layout, and history tracking.

Creates unique experiment IDs, snapshots configuration and environment
metadata, and appends lightweight records to ``history.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentRecord:
    """Lightweight index entry for a created experiment.

    Attributes:
        experiment_id: Unique experiment identifier.
        name: Human-readable experiment name from config.
        created_at: ISO-8601 UTC timestamp.
        output_dir: Absolute path to the experiment output directory.
        config_snapshot: Relative path to the frozen config snapshot.
        env_snapshot: Relative path to the environment JSON snapshot.
    """

    experiment_id: str
    name: str
    created_at: str
    output_dir: str
    config_snapshot: str
    env_snapshot: str
    extra: dict[str, Any] = field(default_factory=dict)


class ExperimentManager:
    """Manage experiment directories, IDs, snapshots, and history.

    Directory layout per experiment::

        {root}/{experiment_id}/
            config.snapshot.yaml
            env.json
            metrics/
            figures/
            reports/
            logs/
    """

    HISTORY_FILENAME = "history.jsonl"

    def __init__(self, root_dir: str | Path = "experiments") -> None:
        """Initialize the manager.

        Args:
            root_dir: Root directory that stores all experiment outputs.
        """
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_id(self, name: str, *, seed: int | None = None) -> str:
        """Create a unique experiment identifier.

        Format: ``{utc_timestamp}_{sanitized_name}_{short_hash}``.

        Args:
            name: Experiment name from configuration.
            seed: Optional seed mixed into the short hash for stability
                across identical names created in the same second.

        Returns:
            Unique experiment ID string.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sanitized = self._sanitize_name(name)
        digest_source = f"{timestamp}:{sanitized}:{seed if seed is not None else ''}"
        short_hash = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:8]
        return f"{timestamp}_{sanitized}_{short_hash}"

    def create_experiment(
        self,
        config: dict[str, Any],
        environment: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        """Allocate an experiment ID, create directories, and snapshot inputs.

        Args:
            config: Normalized experiment configuration.
            environment: Optional environment metadata dictionary to
                persist as ``env.json``.

        Returns:
            :class:`ExperimentRecord` describing the created experiment.

        Raises:
            KeyError: If ``experiment.name`` is missing from ``config``.
        """
        experiment = config.get("experiment")
        if not isinstance(experiment, dict) or "name" not in experiment:
            raise KeyError("config must contain experiment.name")

        name = str(experiment["name"])
        seed = experiment.get("seed")
        experiment_id = self.create_experiment_id(
            name,
            seed=seed if isinstance(seed, int) else None,
        )
        output_dir = self.root_dir / experiment_id
        for subdir in ("metrics", "figures", "reports", "logs"):
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

        config_path = output_dir / "config.snapshot.yaml"
        env_path = output_dir / "env.json"

        snapshot = {k: v for k, v in config.items() if k != "_meta"}
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(snapshot, handle, sort_keys=False)

        env_payload = environment or {}
        with env_path.open("w", encoding="utf-8") as handle:
            json.dump(env_payload, handle, indent=2, default=str)
            handle.write("\n")

        created_at = datetime.now(timezone.utc).isoformat()
        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            created_at=created_at,
            output_dir=str(output_dir),
            config_snapshot=str(config_path.relative_to(self.root_dir)),
            env_snapshot=str(env_path.relative_to(self.root_dir)),
        )
        self.append_history(record)
        return record

    def append_history(self, record: ExperimentRecord) -> Path:
        """Append an experiment record to the root history index.

        Args:
            record: Experiment record to append.

        Returns:
            Path to the history file.
        """
        history_path = self.root_dir / self.HISTORY_FILENAME
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), default=str) + "\n")
        return history_path

    def list_history(self) -> list[ExperimentRecord]:
        """Read all experiment records from the history index.

        Returns:
            List of :class:`ExperimentRecord` instances (possibly empty).
        """
        history_path = self.root_dir / self.HISTORY_FILENAME
        if not history_path.is_file():
            return []

        records: list[ExperimentRecord] = []
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(ExperimentRecord(**payload))
        return records

    def experiment_path(self, experiment_id: str) -> Path:
        """Return the absolute path for an experiment ID.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            Absolute path under the experiments root.
        """
        return self.root_dir / experiment_id

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize an experiment name for filesystem-safe IDs.

        Args:
            name: Raw experiment name.

        Returns:
            Lowercase alphanumeric name with underscores.
        """
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_") or "experiment"
