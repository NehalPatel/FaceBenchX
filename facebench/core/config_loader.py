"""YAML experiment configuration loading and normalization.

Milestone M1 provides a minimal, validated config skeleton. Dataset
loaders and model adapters are intentionally out of scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from facebench.core.registry import CategoryRegistry, get_default_registry


class ConfigError(ValueError):
    """Raised when an experiment configuration is missing or invalid."""


class ConfigLoader:
    """Load, validate, and normalize FaceBench YAML experiment configs.

    The loader performs structural checks required for Milestone M1 and
    expands ``datasets: all`` using the category registry. It does not
    load images, weights, or run evaluations.
    """

    def __init__(self, registry: CategoryRegistry | None = None) -> None:
        """Initialize the loader.

        Args:
            registry: Dataset category registry used to expand
                ``datasets: all``. Defaults to the built-in public
                dataset registry.
        """
        self._registry = registry or get_default_registry()

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a YAML config file and return a normalized dictionary.

        Args:
            path: Path to a YAML experiment configuration file.

        Returns:
            Normalized configuration dictionary.

        Raises:
            ConfigError: If the file cannot be read or fails validation.
        """
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

        if raw is None:
            raise ConfigError(f"Configuration file is empty: {config_path}")
        if not isinstance(raw, dict):
            raise ConfigError(
                f"Configuration root must be a mapping, got {type(raw).__name__}"
            )

        return self.normalize(raw, source_path=config_path)

    def normalize(
        self,
        config: dict[str, Any],
        *,
        source_path: Path | None = None,
    ) -> dict[str, Any]:
        """Validate and normalize a raw configuration mapping.

        Args:
            config: Raw configuration dictionary.
            source_path: Optional path of the source YAML file, stored
                under ``_meta.source_path`` for provenance.

        Returns:
            A shallow-copied, normalized configuration dictionary.

        Raises:
            ConfigError: If required sections or fields are missing.
        """
        normalized = dict(config)

        experiment = normalized.get("experiment")
        if not isinstance(experiment, dict):
            raise ConfigError("Missing required section: 'experiment'")
        if not str(experiment.get("name", "")).strip():
            raise ConfigError("experiment.name must be a non-empty string")

        output_dir = experiment.get("output_dir", "experiments")
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ConfigError("experiment.output_dir must be a non-empty string")
        experiment["output_dir"] = output_dir
        experiment.setdefault("seed", 42)
        normalized["experiment"] = experiment

        device = normalized.get("device", "cpu")
        if not isinstance(device, str) or not device.strip():
            raise ConfigError("'device' must be a non-empty string")
        normalized["device"] = device

        batch_size = normalized.get("batch_size", 1)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigError("'batch_size' must be a positive integer")
        normalized["batch_size"] = batch_size

        normalized["datasets"] = self._normalize_datasets(normalized)
        normalized["models"] = self._normalize_models(normalized)

        matching = normalized.get("matching", {})
        if matching is None:
            matching = {}
        if not isinstance(matching, dict):
            raise ConfigError("'matching' must be a mapping when provided")
        matching.setdefault("method", "cosine")
        matching.setdefault("threshold", 0.4)
        method = matching["method"]
        if method not in {"cosine", "euclidean"}:
            raise ConfigError(
                "matching.method must be 'cosine' or 'euclidean', " f"got {method!r}"
            )
        normalized["matching"] = matching

        evaluation = normalized.get("evaluation", {})
        if evaluation is None:
            evaluation = {}
        if not isinstance(evaluation, dict):
            raise ConfigError("'evaluation' must be a mapping when provided")
        evaluation.setdefault("mode", "verification")
        mode = str(evaluation["mode"]).strip().lower()
        if mode not in {"verification", "identification"}:
            raise ConfigError(
                "evaluation.mode must be 'verification' or 'identification', "
                f"got {evaluation['mode']!r}"
            )
        evaluation["mode"] = mode
        evaluation.setdefault("aggregate_report", False)
        normalized["evaluation"] = evaluation

        normalized["robustness"] = self._normalize_robustness(
            normalized.get("robustness")
        )
        normalized["scalability"] = self._normalize_scalability(
            normalized.get("scalability")
        )
        normalized["detection"] = self._normalize_detection(
            normalized.get("detection")
        )

        normalized["_meta"] = {
            "source_path": str(source_path) if source_path else None,
        }
        return normalized

    def _normalize_datasets(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize single-dataset, multi-dataset, or batch-all forms.

        Args:
            config: Raw configuration dictionary.

        Returns:
            A list of ``{"name": str, "root_path": str | None}`` entries.

        Raises:
            ConfigError: If dataset configuration is invalid.
        """
        if "datasets" in config and config["datasets"] is not None:
            datasets_value = config["datasets"]
            roots = config.get("dataset_roots", {})
            if datasets_value == "all":
                if not isinstance(roots, dict):
                    raise ConfigError(
                        "dataset_roots must be a mapping when datasets: all"
                    )
                entries: list[dict[str, Any]] = []
                for name in self._registry.list_all():
                    root = roots.get(name)
                    entries.append(
                        {
                            "name": name,
                            "root_path": str(root) if root is not None else None,
                        }
                    )
                return entries

            if isinstance(datasets_value, list):
                return [self._coerce_dataset_entry(item) for item in datasets_value]

            raise ConfigError("'datasets' must be 'all' or a list of dataset mappings")

        if "dataset" in config and config["dataset"] is not None:
            return [self._coerce_dataset_entry(config["dataset"])]

        # Milestone M1 allows configs without datasets for round-trip tests.
        return []

    def _coerce_dataset_entry(self, item: Any) -> dict[str, Any]:
        """Coerce a dataset entry into a normalized mapping.

        Args:
            item: Raw dataset entry (must be a mapping with ``name``).

        Returns:
            Normalized dataset entry.

        Raises:
            ConfigError: If the entry is malformed or unknown.
        """
        if not isinstance(item, dict):
            raise ConfigError("Each dataset entry must be a mapping")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("dataset.name must be a non-empty string")
        if not self._registry.is_supported(name):
            supported = ", ".join(self._registry.list_all())
            raise ConfigError(f"Unsupported dataset {name!r}. Supported: {supported}")
        root_path = item.get("root_path")
        return {
            "name": name,
            "root_path": str(root_path) if root_path is not None else None,
            "category": self._registry.get_category(name),
        }

    def _normalize_models(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize single-model or multi-model configuration.

        Args:
            config: Raw configuration dictionary.

        Returns:
            A list of model mappings with at least a ``name`` field.
        """
        if "models" in config and config["models"] is not None:
            models_value = config["models"]
            if not isinstance(models_value, list):
                raise ConfigError("'models' must be a list when provided")
            return [self._coerce_model_entry(item) for item in models_value]

        if "model" in config and config["model"] is not None:
            return [self._coerce_model_entry(config["model"])]

        return []

    @staticmethod
    def _coerce_model_entry(item: Any) -> dict[str, Any]:
        """Coerce a model entry into a normalized mapping.

        Args:
            item: Raw model entry.

        Returns:
            Normalized model entry.

        Raises:
            ConfigError: If the entry is malformed.
        """
        if not isinstance(item, dict):
            raise ConfigError("Each model entry must be a mapping")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("model.name must be a non-empty string")
        entry = dict(item)
        entry["name"] = name
        return entry

    @staticmethod
    def _normalize_detection(value: Any) -> dict[str, Any]:
        """Normalize optional shared detection / alignment configuration.

        Default ``backend: none`` preserves Baseline A (vendor/path embed).
        ``backend: retinaface`` enables Baseline B shared crops.
        """
        if value is None:
            return {
                "backend": "none",
                "output_size": 112,
                "det_size": [640, 640],
                "model_name": "buffalo_l",
                "weights_path": None,
                "skip_failed": True,
                "crop_mode": "bbox_margin",
                "bbox_margin": 0.35,
            }
        if not isinstance(value, dict):
            raise ConfigError("'detection' must be a mapping when provided")

        backend = str(value.get("backend", "none")).strip().lower() or "none"
        allowed = {
            "none",
            "off",
            "disabled",
            "passthrough",
            "identity",
            "retinaface",
            "scrfd",
            "insightface",
            "insightface_scrfd",
        }
        if backend not in allowed:
            raise ConfigError(
                "detection.backend must be one of "
                "none|passthrough|retinaface, "
                f"got {backend!r}"
            )

        output_size = int(value.get("output_size", 112))
        if output_size < 32:
            raise ConfigError("detection.output_size must be >= 32")

        det_size = value.get("det_size", [640, 640])
        if not isinstance(det_size, (list, tuple)) or len(det_size) != 2:
            raise ConfigError("detection.det_size must be a [H, W] pair")
        det_h, det_w = int(det_size[0]), int(det_size[1])
        if det_h < 32 or det_w < 32:
            raise ConfigError("detection.det_size entries must be >= 32")

        weights_path = value.get("weights_path")
        if weights_path is not None:
            weights_path = str(weights_path)

        crop_mode = str(value.get("crop_mode", "bbox_margin")).strip().lower()
        crop_mode = crop_mode.replace("-", "_")
        if crop_mode in {"bbox", "margin", "padded", "bbox_padded"}:
            crop_mode = "bbox_margin"
        if crop_mode in {"norm", "aligned", "arcface", "112"}:
            crop_mode = "norm_112"
        if crop_mode not in {"bbox_margin", "norm_112"}:
            raise ConfigError(
                "detection.crop_mode must be 'bbox_margin' or 'norm_112', "
                f"got {crop_mode!r}"
            )

        bbox_margin = float(value.get("bbox_margin", 0.35))
        if bbox_margin < 0.0 or bbox_margin > 2.0:
            raise ConfigError("detection.bbox_margin must be in [0, 2]")

        return {
            "backend": backend,
            "output_size": output_size,
            "det_size": [det_h, det_w],
            "model_name": str(value.get("model_name", "buffalo_l")),
            "weights_path": weights_path,
            "skip_failed": bool(value.get("skip_failed", True)),
            "crop_mode": crop_mode,
            "bbox_margin": bbox_margin,
        }

    @staticmethod
    def _normalize_robustness(value: Any) -> dict[str, Any]:
        """Normalize optional synthetic robustness configuration."""
        if value is None:
            return {"enabled": False, "transforms": []}
        if not isinstance(value, dict):
            raise ConfigError("'robustness' must be a mapping when provided")
        enabled = bool(value.get("enabled", False))
        transforms = value.get("transforms", [])
        if transforms is None:
            transforms = []
        if not isinstance(transforms, list):
            raise ConfigError("robustness.transforms must be a list of names")
        names = [str(item).strip() for item in transforms if str(item).strip()]
        if enabled and not names:
            raise ConfigError(
                "robustness.enabled requires a non-empty robustness.transforms list"
            )
        return {"enabled": enabled, "transforms": names}

    @staticmethod
    def _normalize_scalability(value: Any) -> dict[str, Any]:
        """Normalize optional scalability ladder configuration."""
        if value is None:
            return {"enabled": False, "identity_counts": []}
        if not isinstance(value, dict):
            raise ConfigError("'scalability' must be a mapping when provided")
        enabled = bool(value.get("enabled", False))
        counts = value.get("identity_counts", [10, 100, 500, 1000, 5000])
        if counts is None:
            counts = []
        if not isinstance(counts, list):
            raise ConfigError("scalability.identity_counts must be a list of integers")
        parsed: list[int] = []
        for item in counts:
            try:
                count = int(item)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "scalability.identity_counts entries must be integers, "
                    f"got {item!r}"
                ) from exc
            if count < 1:
                raise ConfigError("scalability.identity_counts must be >= 1")
            parsed.append(count)
        if enabled and not parsed:
            raise ConfigError(
                "scalability.enabled requires a non-empty identity_counts list"
            )
        return {"enabled": enabled, "identity_counts": parsed}


def load_config(
    path: str | Path,
    registry: CategoryRegistry | None = None,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`ConfigLoader.load`.

    Args:
        path: Path to a YAML experiment configuration file.
        registry: Optional dataset category registry override.

    Returns:
        Normalized configuration dictionary.
    """
    return ConfigLoader(registry=registry).load(path)
