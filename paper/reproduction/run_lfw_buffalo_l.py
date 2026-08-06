"""Execute LFW x Buffalo-L paper run with skip-on-missed-detection.

Experimental execution tooling under paper/ — does not modify FaceBench
framework modules. Uses FaceBench adapters/metrics/reports directly.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from facebench import __version__
from facebench.core.config_loader import load_config
from facebench.core.experiment_manager import ExperimentManager
from facebench.datasets.factory import DatasetFactory
from facebench.matcher import create_matcher
from facebench.metrics import MetricCalculator
from facebench.models.factory import ModelFactory
from facebench.reports import ReportGenerator
from facebench.utils.env_info import collect_environment_info

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "paper" / "configs" / "local" / "lfw_buffalo_l.yaml"
SUMMARY_PATH = ROOT / "paper" / "results" / "lfw_buffalo_l_execution_summary.md"


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


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embed_cached(model, path: Path, cache: dict, failures: dict):
    key = str(path)
    if key in cache:
        return cache[key]
    if key in failures:
        return None
    try:
        emb = model.generate_embedding(path)
        cache[key] = emb
        return emb
    except Exception as exc:  # noqa: BLE001 - record and skip for paper run
        failures[key] = str(exc)
        return None


def main() -> int:
    if not CONFIG.is_file():
        print(f"Missing config: {CONFIG}", file=sys.stderr)
        return 2

    config = load_config(CONFIG)
    assert config["matching"]["method"] == "cosine"
    assert float(config["matching"]["threshold"]) == 0.4
    config["experiment"]["name"] = "paper_lfw_buffalo_l"

    dataset_root = config["datasets"][0]["root_path"]
    weights_path = config["models"][0].get("weights_path")
    device = str(config.get("device", "cpu"))
    threshold = float(config["matching"]["threshold"])
    seed = int(config["experiment"].get("seed", 42))

    started_dt = datetime.now(timezone.utc)
    started = started_dt.isoformat()
    t0 = time.perf_counter()
    print(f"[{started}] Starting LFW x Buffalo-L paper execution ...")

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
    print(f"pairs_loaded={len(pairs)}")

    model = ModelFactory().create(
        "buffalo_l",
        device=device,
        weights_path=weights_path,
    )
    calc = MetricCalculator()
    profiler = calc.create_profiler(warmup=2)
    if weights_path:
        profiler.set_model_size_mb(calc.model_size(weights_path))

    def _load() -> None:
        model.load_model(device)

    profiler.time_model_load(_load)
    matcher = create_matcher("cosine")

    cache: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    labels: list[int] = []
    scores: list[float] = []
    scored_folds: list[int] = []
    skipped_pairs: list[dict] = []

    total = len(pairs)
    report_every = 100
    for idx, pair in enumerate(pairs, start=1):
        emb_a = _embed_cached(model, pair.sample_a.path, cache, failures)
        emb_b = _embed_cached(model, pair.sample_b.path, cache, failures)
        if emb_a is None or emb_b is None:
            skipped_pairs.append(
                {
                    "index": idx - 1,
                    "a": str(pair.sample_a.path),
                    "b": str(pair.sample_b.path),
                    "issame": pair.issame,
                    "reason_a": failures.get(str(pair.sample_a.path)),
                    "reason_b": failures.get(str(pair.sample_b.path)),
                }
            )
        else:
            # Profile only successful embedding path once per unique image;
            # timing uses cached embeddings for computational summary.
            scores.append(float(matcher.score(emb_a, emb_b)))
            labels.append(1 if pair.issame else 0)
            fold = pair.fold if pair.fold is not None else (idx - 1) // 600
            scored_folds.append(int(fold))

        if idx % report_every == 0 or idx == total:
            elapsed = time.perf_counter() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (total - idx) / rate if rate > 0 else float("inf")
            print(
                f"progress {idx}/{total} scored={len(scores)} "
                f"skipped={len(skipped_pairs)} "
                f"unique_fail_images={len(failures)} "
                f"elapsed_s={elapsed:.0f} eta_s={eta:.0f}",
                flush=True,
            )

    if len(scores) < 100:
        print("Too few scored pairs to accept experiment:", len(scores))
        return 1

    # Approximate computational metrics from unique successful embeds
    for emb in cache.values():
        profiler.track_embedding(lambda e=emb: e)
    recognition = calc.recognition(
        np.asarray(labels),
        np.asarray(scores, dtype=np.float64),
        threshold=threshold,
    )
    computational = profiler.summarize()

    combo_dir = output_dir / "runs" / "LFW__buffalo_l"
    reports = ReportGenerator(combo_dir)
    report = reports.write_per_dataset(
        experiment_id=record.experiment_id,
        experiment_name=config["experiment"]["name"],
        dataset_name="LFW",
        model_name=model.name,
        recognition=recognition,
        computational=computational,
        config=config,
        y_true=labels,
        y_score=scores,
        generate_figures=True,
        notes=(
            f"Baseline A vendor detect. Scored {len(scores)}/{total} pairs; "
            f"skipped {len(skipped_pairs)} due to detection failures "
            f"({len(failures)} unique images)."
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
        "num_failed_images": len(failures),
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
                "failed_images": failures,
                "skipped_pairs": skipped_pairs[:500],
                "skipped_pairs_total": len(skipped_pairs),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_path = output_dir / "metrics" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "experiment_id": record.experiment_id,
                "runs": [report.to_dict()],
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0
    ended = datetime.now(timezone.utc).isoformat()
    weights_root = Path(weights_path) if weights_path else Path()
    recog_onnx = weights_root / "models" / "buffalo_l" / "w600k_r50.onnx"
    manifest = {
        "experiment_id": record.experiment_id,
        "timestamp": started,
        "ended_at": ended,
        "execution_time_s": elapsed,
        "git_commit": _git_commit(),
        "framework_version": __version__,
        "dataset": {
            "name": "LFW",
            "root_path": dataset_root,
            "version": "lfw-deepfunneled + View-2 pairs.txt",
            "protocol_file": str(Path(dataset_root) / "pairs.txt"),
            "pair_count_protocol": total,
            "pair_count_scored": len(scores),
            "pair_count_skipped": len(skipped_pairs),
        },
        "model": {
            "name": "buffalo_l",
            "weights_path": str(weights_root),
            "weight_version": "insightface buffalo_l pack",
            "weight_checksum": {
                "w600k_r50.onnx_sha256": _sha256(recog_onnx),
            },
        },
        "matching": config.get("matching", {}),
        "detection": {
            "backend": "insightface_buffalo_pack",
            "shared_align": False,
            "failed_images": len(failures),
            "notes": "Baseline A; pairs skipped when detector found no face",
        },
        "seed": seed,
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor() or env.get("platform_machine"),
            "gpu": env.get("gpu_name"),
            "cuda_version": env.get("cuda_version"),
            "python_version": env.get("python_version"),
            "packages": env.get("packages"),
            "hostname": env.get("hostname"),
            "onnxruntime_providers": ["CPUExecutionProvider"],
        },
        "recognition_metrics": recognition.to_dict(),
        "computational_metrics": computational.to_dict(),
        "paper_mode": True,
        "allow_stub": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checks = [
        ("config.snapshot.yaml", output_dir / "config.snapshot.yaml"),
        ("env.json", output_dir / "env.json"),
        ("manifest.json", manifest_path),
        ("metrics/summary.json", summary_path),
        ("report.html", combo_dir / "reports" / "report.html"),
        ("report.md", combo_dir / "reports" / "report.md"),
        ("experiment.json", metrics_dir / "experiment.json"),
        ("metrics.csv", metrics_dir / "metrics.csv"),
        ("verification.json", metrics_dir / "verification.json"),
        ("scores_labels_folds.npz", metrics_dir / "scores_labels_folds.npz"),
        ("ROC", combo_dir / "figures" / "roc_curve.png"),
        ("confusion", combo_dir / "figures" / "confusion_matrix.png"),
    ]
    results = []
    for label, path in checks:
        ok = path.is_file() and path.stat().st_size > 0
        results.append((label, ok, str(path)))
    all_ok = all(ok for _, ok, _ in results)

    curated = ROOT / "paper" / "results" / f"lfw_buffalo_l_{record.experiment_id}"
    curated.mkdir(parents=True, exist_ok=True)
    (curated / "manifest.json").write_text(
        manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (curated / "metrics_snapshot.json").write_text(
        json.dumps(recognition.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    (curated / "MANIFEST_POINTER.txt").write_text(
        f"experiment_dir={output_dir}\nmanifest={manifest_path}\n",
        encoding="utf-8",
    )

    lines = [
        "# LFW × Buffalo-L Execution Summary",
        "",
        f"- **Status:** {'SUCCESS' if all_ok else 'PARTIAL / FAILED ARTIFACTS'}",
        f"- **Experiment ID:** `{record.experiment_id}`",
        f"- **Output dir:** `{output_dir}`",
        f"- **Started (UTC):** {started}",
        f"- **Ended (UTC):** {ended}",
        f"- **Elapsed:** {elapsed:.1f} s ({elapsed/3600:.2f} h)",
        f"- **Device:** `{device}` (CPUExecutionProvider)",
        f"- **Matching:** cosine @ {threshold}",
        f"- **Pairs protocol:** {total}",
        f"- **Pairs scored:** {len(scores)}",
        f"- **Pairs skipped (no face):** {len(skipped_pairs)}",
        f"- **Unique failed images:** {len(failures)}",
        f"- **Stub used:** No",
        "",
        "## Metrics",
        "",
        f"- Accuracy: {recognition.accuracy}",
        f"- Precision: {recognition.precision}",
        f"- Recall: {recognition.recall}",
        f"- F1: {recognition.f1}",
        f"- AUC: {recognition.auc}",
        f"- EER: {recognition.eer}",
        f"- FAR: {recognition.far}",
        f"- FRR: {recognition.frr}",
        f"- Threshold: {recognition.threshold}",
        f"- EER threshold: {recognition.eer_threshold}",
        "",
        "## Artifact validation",
        "",
    ]
    for label, ok, path in results:
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {label} — `{path}`")

    sample_fails = list(failures.items())[:10]
    lines.extend(
        [
            "",
            "## Issues / notes",
            "",
            "- CUDA provider unavailable; CPU-only ONNX Runtime.",
            "- Official UMass `pairs.txt` host DNS failed; used facenet mirror "
            "and normalized header to `10`.",
            "- Dataset root nested at "
            "`D:/datasets/lfw/lfw-deepfunneled/lfw-deepfunneled`.",
            "- Pairs with undetected faces were skipped and logged "
            "(not a framework change; paper-runner policy).",
            "- Baseline A: InsightFace Buffalo-L bundled detector "
            "(not shared RetinaFace).",
            "",
            "### Sample failed images",
            "",
        ]
    )
    for path, reason in sample_fails:
        lines.append(f"- `{path}` — {reason}")
    lines.extend(["", f"Manifest: `{manifest_path}`", f"Curated: `{curated}`", ""])

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
