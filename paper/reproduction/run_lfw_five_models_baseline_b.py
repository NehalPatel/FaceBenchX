"""Execute LFW x five models under frozen Baseline B shared RetinaFace.

Paper execution tooling — does not modify FaceBench framework modules.
Uses FaceBench adapters, metrics, and reports directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facebench import __version__  # noqa: E402
from facebench.core.experiment_manager import ExperimentManager  # noqa: E402
from facebench.datasets.factory import DatasetFactory  # noqa: E402
from facebench.detection import as_image_transform, create_aligner  # noqa: E402
from facebench.detection.align import FaceDetectionError  # noqa: E402
from facebench.matcher import create_matcher  # noqa: E402
from facebench.metrics import MetricCalculator  # noqa: E402
from facebench.models.factory import ModelFactory  # noqa: E402
from facebench.models.imaging import load_image_rgb  # noqa: E402
from facebench.reports import ReportGenerator  # noqa: E402
from facebench.utils.env_info import collect_environment_info  # noqa: E402

from paper.reproduction.arch.loaders import (  # noqa: E402
    build_adaface_ir50,
    build_magface_iresnet50,
)

DEFAULT_CONFIG = ROOT / "paper" / "configs" / "local" / "lfw_five_models_baseline_b.yaml"
SUMMARY_PATH = ROOT / "paper" / "results" / "lfw_five_models_baseline_b_report.md"
WEIGHTS_LEDGER = ROOT / "paper" / "weights" / "checksums.sha256"


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize single-model or multi-model paper YAML into a common shape."""
    config = dict(raw)
    if "models" not in config:
        model = config.get("model")
        if not isinstance(model, dict):
            raise ValueError("Config requires models: list or model: mapping")
        config["models"] = [model]
    if "dataset" in config and "datasets" not in config:
        config["datasets"] = [config["dataset"]]
    return config


def _unique_paths(pairs) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for pair in pairs:
        for sample in (pair.sample_a, pair.sample_b):
            key = str(sample.path)
            if key not in seen:
                seen.add(key)
                ordered.append(Path(sample.path))
    return ordered


def _build_aligned_cache(
    paths: list[Path],
    transform,
    *,
    report_every: int = 200,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Detect/align once; share crops across all five models."""
    cache: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    t0 = time.perf_counter()
    total = len(paths)
    for idx, path in enumerate(paths, start=1):
        key = str(path)
        try:
            rgb = load_image_rgb(path)
            cache[key] = transform(rgb)
        except Exception as exc:  # noqa: BLE001 - record and continue
            failures[key] = str(exc)
        if idx % report_every == 0 or idx == total:
            elapsed = time.perf_counter() - t0
            print(
                f"align {idx}/{total} ok={len(cache)} fail={len(failures)} "
                f"elapsed_s={elapsed:.0f}",
                flush=True,
            )
    return cache, failures


def _create_model(spec: dict[str, Any], device: str):
    name = str(spec["name"])
    factory = ModelFactory()
    kwargs: dict[str, Any] = {"allow_stub": False}
    if name.lower() in {"dlib", "dlib_fr"}:
        weights_dir = spec.get("weights_dir") or spec.get("weights_path")
        if weights_dir:
            kwargs["weights_path"] = weights_dir
        return factory.create(name, device=device, **kwargs)
    if name.lower() == "adaface":
        # Inject a fully loaded backbone; omit weights_path so the adapter
        # does not reload Lightning keys into the wrapped module.
        weights = Path(spec["weights_path"])
        model = build_adaface_ir50(weights, device=device)
        return factory.create(
            name,
            device=device,
            model=model,
            allow_stub=False,
        )
    if name.lower() == "magface":
        weights = Path(spec["weights_path"])
        model = build_magface_iresnet50(weights, device=device)
        return factory.create(
            name,
            device=device,
            model=model,
            allow_stub=False,
        )
    weights_path = spec.get("weights_path")
    if weights_path:
        kwargs["weights_path"] = weights_path
    return factory.create(name, device=device, **kwargs)


def _embed_cached(
    model,
    path: Path,
    aligned: dict[str, np.ndarray],
    align_failures: dict[str, str],
    emb_cache: dict[str, np.ndarray],
    emb_failures: dict[str, str],
):
    key = str(path)
    if key in emb_cache:
        return emb_cache[key]
    if key in align_failures:
        emb_failures[key] = f"align_failed: {align_failures[key]}"
        return None
    if key in emb_failures:
        return None
    crop = aligned.get(key)
    if crop is None:
        emb_failures[key] = "missing_aligned_crop"
        return None
    try:
        emb = model.generate_embedding(crop)
        emb_cache[key] = emb
        return emb
    except Exception as exc:  # noqa: BLE001
        emb_failures[key] = str(exc)
        return None


def _weight_checksum_payload(spec: dict[str, Any]) -> dict[str, Any]:
    name = str(spec["name"]).lower()
    payload: dict[str, Any] = {"name": name}
    if name == "dlib":
        weights_dir = Path(spec.get("weights_dir") or spec.get("weights_path") or "")
        pred = weights_dir / "shape_predictor_5_face_landmarks.dat"
        rec = weights_dir / "dlib_face_recognition_resnet_model_v1.dat"
        payload["paths"] = {
            "predictor": str(pred),
            "recognition": str(rec),
        }
        payload["sha256"] = {
            pred.name: _sha256(pred),
            rec.name: _sha256(rec),
        }
        return payload
    if name == "buffalo_l":
        root = Path(spec.get("weights_path") or "")
        onnx = root / "models" / "buffalo_l" / "w600k_r50.onnx"
        payload["paths"] = {"recognition_onnx": str(onnx)}
        payload["sha256"] = {onnx.name: _sha256(onnx)}
        return payload
    if name == "facenet":
        payload["paths"] = {"weights": "facenet-pytorch pretrained=vggface2 (auto)"}
        payload["sha256"] = None
        return payload
    path = Path(spec["weights_path"])
    payload["paths"] = {"weights": str(path)}
    payload["sha256"] = {path.name: _sha256(path)}
    return payload


def _append_checksum_ledger(entries: list[dict[str, Any]]) -> None:
    WEIGHTS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FaceBench paper weight checksums (Baseline B five-model LFW)",
        f"# generated_utc={datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for entry in entries:
        sha = entry.get("sha256")
        if not isinstance(sha, dict):
            lines.append(f"# {entry.get('name')}: {entry.get('paths')}")
            continue
        for filename, digest in sha.items():
            if digest:
                lines.append(f"{digest}  {filename}")
    WEIGHTS_LEDGER.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_model(
    *,
    model_spec: dict[str, Any],
    pairs,
    aligned: dict[str, np.ndarray],
    align_failures: dict[str, str],
    config: dict[str, Any],
    output_dir: Path,
    experiment_id: str,
    device: str,
    threshold: float,
) -> tuple[Any, dict[str, Any]]:
    model_name = str(model_spec["name"])
    print(f"\n=== model={model_name} device={device} ===", flush=True)
    t0 = time.perf_counter()
    model = _create_model(model_spec, device)
    calc = MetricCalculator()
    profiler = calc.create_profiler(warmup=2)

    def _load() -> None:
        model.load_model(device)

    profiler.time_model_load(_load)
    matcher = create_matcher("cosine")

    emb_cache: dict[str, np.ndarray] = {}
    emb_failures: dict[str, str] = {}
    labels: list[int] = []
    scores: list[float] = []
    scored_folds: list[int] = []
    skipped_pairs: list[dict[str, Any]] = []

    total = len(pairs)
    report_every = 200
    for idx, pair in enumerate(pairs, start=1):
        emb_a = _embed_cached(
            model,
            pair.sample_a.path,
            aligned,
            align_failures,
            emb_cache,
            emb_failures,
        )
        emb_b = _embed_cached(
            model,
            pair.sample_b.path,
            aligned,
            align_failures,
            emb_cache,
            emb_failures,
        )
        if emb_a is None or emb_b is None:
            skipped_pairs.append(
                {
                    "index": idx - 1,
                    "a": str(pair.sample_a.path),
                    "b": str(pair.sample_b.path),
                    "issame": pair.issame,
                    "reason_a": emb_failures.get(str(pair.sample_a.path)),
                    "reason_b": emb_failures.get(str(pair.sample_b.path)),
                }
            )
        else:
            scores.append(float(matcher.score(emb_a, emb_b)))
            labels.append(1 if pair.issame else 0)
            fold = pair.fold if pair.fold is not None else (idx - 1) // 600
            scored_folds.append(int(fold))

        if idx % report_every == 0 or idx == total:
            elapsed = time.perf_counter() - t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (total - idx) / rate if rate > 0 else float("inf")
            print(
                f"{model_name} {idx}/{total} scored={len(scores)} "
                f"skipped={len(skipped_pairs)} elapsed_s={elapsed:.0f} eta_s={eta:.0f}",
                flush=True,
            )

    min_scored = min(100, max(10, total // 2))
    if len(scores) < min_scored:
        raise RuntimeError(
            f"{model_name}: too few scored pairs ({len(scores)} < {min_scored}); "
            "aborting model run"
        )

    for emb in emb_cache.values():
        profiler.track_embedding(lambda e=emb: e)

    recognition = calc.recognition(
        np.asarray(labels),
        np.asarray(scores, dtype=np.float64),
        threshold=threshold,
    )
    computational = profiler.summarize()

    combo_dir = output_dir / "runs" / f"LFW__{model.name}"
    reports = ReportGenerator(combo_dir)
    report = reports.write_per_dataset(
        experiment_id=experiment_id,
        experiment_name=str(config["experiment"]["name"]),
        dataset_name="LFW",
        model_name=model.name,
        recognition=recognition,
        computational=computational,
        config=config,
        y_true=labels,
        y_score=scores,
        generate_figures=True,
        notes=(
            f"Baseline B shared RetinaFace bbox_margin. "
            f"Scored {len(scores)}/{total}; skipped {len(skipped_pairs)} "
            f"(align_fail_images={len(align_failures)}, "
            f"embed_fail_images={len(emb_failures)})."
        ),
    )

    metrics_dir = combo_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        metrics_dir / "scores_labels_folds.npz",
        y_true=np.asarray(labels, dtype=np.int8),
        y_score=np.asarray(scores, dtype=np.float64),
        folds=np.asarray(scored_folds, dtype=np.int16),
    )
    verification_payload = {
        "num_pairs_protocol": total,
        "num_pairs_scored": len(scores),
        "num_pairs_skipped": len(skipped_pairs),
        "num_align_failed_images": len(align_failures),
        "num_embed_failed_images": len(emb_failures),
        "recognition": recognition.to_dict(),
        "computational": computational.to_dict(),
    }
    (metrics_dir / "verification.json").write_text(
        json.dumps(verification_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "skipped_pairs.json").write_text(
        json.dumps(
            {
                "align_failed_images": align_failures,
                "embed_failed_images": emb_failures,
                "skipped_pairs": skipped_pairs[:500],
                "skipped_pairs_total": len(skipped_pairs),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    model_manifest = {
        "model": _weight_checksum_payload(model_spec),
        "recognition_metrics": recognition.to_dict(),
        "computational_metrics": computational.to_dict(),
        "pairs_scored": len(scores),
        "pairs_skipped": len(skipped_pairs),
        "execution_time_s": time.perf_counter() - t0,
    }
    (combo_dir / "model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{model_name} done acc={recognition.accuracy:.4f} "
        f"auc={recognition.auc:.4f} eer={recognition.eer:.4f} "
        f"scored={len(scores)}/{total}",
        flush=True,
    )
    return report, model_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to local five-model Baseline B YAML",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap for smoke tests (overrides config)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model subset (default: all five)",
    )
    args = parser.parse_args(argv)

    if not args.config.is_file():
        print(f"Missing config: {args.config}", file=sys.stderr)
        return 2

    raw = _load_yaml(args.config)
    config = _normalize_config(raw)
    assert config["matching"]["method"] == "cosine"
    assert abs(float(config["matching"]["threshold"]) - 0.4) < 1e-9
    det = config.get("detection") or {}
    assert str(det.get("backend")) == "retinaface"
    assert str(det.get("crop_mode")) == "bbox_margin"

    model_specs: list[dict[str, Any]] = list(config["models"])
    if args.models:
        wanted = {m.strip().lower() for m in args.models.split(",") if m.strip()}
        model_specs = [m for m in model_specs if str(m["name"]).lower() in wanted]
        if not model_specs:
            print(f"No models matched --models={args.models}", file=sys.stderr)
            return 2

    dataset_root = config["datasets"][0]["root_path"]
    device = str(config.get("device", "cpu"))
    threshold = float(config["matching"]["threshold"])
    seed = int(config["experiment"].get("seed", 42))
    max_pairs = args.max_pairs
    if max_pairs is None:
        cfg_max = config.get("evaluation", {}).get("max_pairs")
        max_pairs = int(cfg_max) if cfg_max is not None else None

    started_dt = datetime.now(timezone.utc)
    started = started_dt.isoformat()
    t0 = time.perf_counter()
    print(f"[{started}] Starting LFW x five-model Baseline B ...")
    print(f"config={args.config}")
    print(f"models={[m['name'] for m in model_specs]} device={device}")

    env = collect_environment_info().to_dict()
    manager = ExperimentManager(root_dir=config["experiment"]["output_dir"])
    record = manager.create_experiment(config, environment=env)
    output_dir = Path(record.output_dir)
    print(f"experiment_id={record.experiment_id}")
    print(f"output_dir={output_dir}")

    dataset = DatasetFactory().create("LFW", dataset_root)
    integrity = dataset.validate_integrity()
    if not integrity.ok:
        print("Integrity failed:", integrity.messages)
        return 1
    pairs = dataset.load_identity_pairs()
    if max_pairs is not None:
        pairs = pairs[: max_pairs]
    print(f"pairs_loaded={len(pairs)}")

    aligner = create_aligner(det, device=device)
    transform = as_image_transform(aligner)
    unique = _unique_paths(pairs)
    print(f"unique_images={len(unique)} — shared RetinaFace pass")
    aligned, align_failures = _build_aligned_cache(unique, transform)
    print(
        f"align_complete ok={len(aligned)} fail={len(align_failures)}",
        flush=True,
    )

    reports = []
    model_manifests = []
    checksum_entries = []
    for spec in model_specs:
        report, model_manifest = run_one_model(
            model_spec=spec,
            pairs=pairs,
            aligned=aligned,
            align_failures=align_failures,
            config=config,
            output_dir=output_dir,
            experiment_id=record.experiment_id,
            device=device,
            threshold=threshold,
        )
        reports.append(report)
        model_manifests.append(model_manifest)
        checksum_entries.append(model_manifest["model"])

    agg = ReportGenerator(output_dir)
    rows = agg.write_aggregated(reports, also_write_comparison_figures=True)

    summary_path = output_dir / "metrics" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "experiment_id": record.experiment_id,
                "protocol": "Baseline B shared RetinaFace bbox_margin",
                "runs": [r.to_dict() for r in reports],
                "comparison_rows": [row.to_dict() for row in rows],
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0
    ended = datetime.now(timezone.utc).isoformat()
    manifest = {
        "experiment_id": record.experiment_id,
        "timestamp": started,
        "ended_at": ended,
        "execution_time_s": elapsed,
        "git_commit": _git_commit(),
        "framework_version": __version__,
        "protocol": {
            "name": "Baseline B",
            "detection": det,
            "matching": config.get("matching", {}),
            "shared_align_cache": True,
            "align_failed_images": len(align_failures),
            "unique_images": len(unique),
            "pair_count_protocol": len(pairs),
        },
        "dataset": {
            "name": "LFW",
            "root_path": dataset_root,
            "version": "lfw-deepfunneled + View-2 pairs.txt",
        },
        "models": model_manifests,
        "seed": seed,
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor() or env.get("platform_machine"),
            "gpu": env.get("gpu_name"),
            "cuda_version": env.get("cuda_version"),
            "python_version": env.get("python_version"),
            "packages": env.get("packages"),
            "hostname": env.get("hostname"),
        },
        "paper_mode": True,
        "allow_stub": False,
        "artifacts": {
            "aggregated_md": str(output_dir / "aggregated" / "comparison.md"),
            "aggregated_html": str(output_dir / "aggregated" / "comparison.html"),
            "aggregated_csv": str(output_dir / "aggregated" / "comparison.csv"),
            "summary_json": str(summary_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _append_checksum_ledger(checksum_entries)

    curated = ROOT / "paper" / "results" / f"lfw_five_models_{record.experiment_id}"
    curated.mkdir(parents=True, exist_ok=True)
    (curated / "manifest.json").write_text(
        manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (curated / "comparison.md").write_text(
        (output_dir / "aggregated" / "comparison.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (curated / "comparison.csv").write_text(
        (output_dir / "aggregated" / "comparison.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (curated / "MANIFEST_POINTER.txt").write_text(
        f"experiment_dir={output_dir}\nmanifest={manifest_path}\n",
        encoding="utf-8",
    )

    # Single comparison report under paper/results
    lines = [
        "# LFW × Five Models — Baseline B Comparison Report",
        "",
        f"- **Status:** SUCCESS",
        f"- **Experiment ID:** `{record.experiment_id}`",
        f"- **Output dir:** `{output_dir}`",
        f"- **Started (UTC):** {started}",
        f"- **Ended (UTC):** {ended}",
        f"- **Elapsed:** {elapsed:.1f} s ({elapsed/3600:.2f} h)",
        f"- **Device:** `{device}`",
        f"- **Matching:** cosine @ {threshold}",
        f"- **Detection:** RetinaFace/SCRFD shared, `crop_mode=bbox_margin`, "
        f"margin={det.get('bbox_margin')}",
        f"- **Pairs:** {len(pairs)} (scored per model may differ when embeds fail)",
        f"- **Unique images aligned:** {len(aligned)} / {len(unique)} "
        f"(failed={len(align_failures)})",
        f"- **Stub used:** No",
        "",
        "## Per-model metrics",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | AUC | EER | FAR | FRR | "
        "Scored | Skipped |",
        "|-------|----------|-----------|--------|----|-----|-----|-----|-----|"
        "--------|---------|",
    ]
    for report, mm in zip(reports, model_manifests):
        rec = report.recognition
        assert rec is not None
        lines.append(
            f"| {report.model_name} | {rec.accuracy:.4f} | {rec.precision:.4f} | "
            f"{rec.recall:.4f} | {rec.f1:.4f} | {rec.auc:.4f} | {rec.eer:.4f} | "
            f"{rec.far:.4f} | {rec.frr:.4f} | {mm['pairs_scored']} | "
            f"{mm['pairs_skipped']} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Manifest: `{manifest_path}`",
            f"- Aggregated MD: `{output_dir / 'aggregated' / 'comparison.md'}`",
            f"- Aggregated HTML: `{output_dir / 'aggregated' / 'comparison.html'}`",
            f"- Aggregated CSV: `{output_dir / 'aggregated' / 'comparison.csv'}`",
            f"- Weight checksums: `{WEIGHTS_LEDGER}`",
            f"- Curated copy: `{curated}`",
            "",
            "## Protocol notes",
            "",
            "- Shared RetinaFace alignment ran once; crops reused across models.",
            "- AdaFace/MagFace architectures injected via `paper/reproduction/arch` "
            "(framework adapters unchanged).",
            "- AdaFace uses RGB->BGR channel flip for official [-1,1] inputs; "
            "MagFace uses BGR [0,1] input adaptation to match official MagFace inference.",
            "- FaceNet uses facenet-pytorch VGGFace2 pretrained weights.",
            "- Dlib uses official 5-point predictor + ResNet recognition `.dat` files.",
            "- Buffalo-L uses InsightFace pack with vendor re-detect on shared crop.",
            "",
        ]
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (curated / "comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
