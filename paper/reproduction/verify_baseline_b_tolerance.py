"""Compare Baseline A vs Baseline B on the same LFW pairs (Buffalo-L).

Does not modify recognition adapters. Uses shared RetinaFaceAligner for B
and vendor-path embeds for A. Writes a tolerance report under paper/results/.

Usage:
  python paper/reproduction/verify_baseline_b_tolerance.py
  python paper/reproduction/verify_baseline_b_tolerance.py --max-pairs 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from facebench.core.config_loader import load_config
from facebench.datasets.factory import DatasetFactory
from facebench.detection import as_image_transform, create_aligner
from facebench.evaluation.verification import run_verification
from facebench.matcher import create_matcher
from facebench.metrics import MetricCalculator
from facebench.models.factory import ModelFactory

ROOT = Path(__file__).resolve().parents[2]
LOCAL_A = ROOT / "paper" / "configs" / "local" / "lfw_buffalo_l.yaml"
REPORT = ROOT / "paper" / "results" / "baseline_b_tolerance_report.md"

# From docs/BaselineB-RetinaFace-Migration-Plan.md
TOL_ACC = 0.015
TOL_AUC = 0.015
TOL_EER = 0.020

# Published full Baseline A reference (not required to match on a subset)
REF_A = {
    "experiment_id": "20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c",
    "accuracy": 0.9817,
    "auc": 0.9892,
    "eer": 0.0238,
}


def _run(pairs, model, matcher, calc, image_transform, transform_name, threshold, skip):
    profiler = calc.create_profiler(warmup=1)
    profiler.time_model_load(lambda: None)
    return run_verification(
        pairs,
        model,
        matcher,
        profiler,
        threshold=threshold,
        metrics=calc,
        image_transform=image_transform,
        transform_name=transform_name,
        skip_failed_detections=skip,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--config", type=Path, default=LOCAL_A)
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Missing local config: {args.config}", file=sys.stderr)
        print("Copy paper/configs/lfw_buffalo_l.yaml to paper/configs/local/ first.")
        return 2

    config = load_config(args.config)
    dataset_root = config["datasets"][0]["root_path"]
    weights = config["models"][0].get("weights_path")
    device = str(config.get("device", "cpu"))
    threshold = float(config["matching"]["threshold"])

    dataset = DatasetFactory().create("LFW", dataset_root)
    integrity = dataset.validate_integrity()
    if not integrity.ok:
        print("Integrity failed:", integrity.messages)
        return 1

    all_pairs = dataset.load_identity_pairs()
    # LFW View-2 fold layout: 300 same then 300 different per fold.
    # Take a balanced subset so AUC/EER are meaningful.
    n = max(0, int(args.max_pairs))
    half = n // 2
    same = [pair for pair in all_pairs if pair.issame][:half]
    diff = [pair for pair in all_pairs if not pair.issame][: n - len(same)]
    pairs = same + diff
    n_same = sum(1 for pair in pairs if pair.issame)
    n_diff = len(pairs) - n_same
    print(
        f"pairs={len(pairs)} same={n_same} diff={n_diff} "
        f"device={device} threshold={threshold}"
    )

    model = ModelFactory().create("buffalo_l", device=device, weights_path=weights)
    model.load_model(device)
    matcher = create_matcher("cosine")
    calc = MetricCalculator()

    # Baseline A
    t0 = time.perf_counter()
    result_a = _run(pairs, model, matcher, calc, None, None, threshold, True)
    time_a = time.perf_counter() - t0
    print(
        f"Baseline A: acc={result_a.recognition.accuracy:.4f} "
        f"auc={result_a.recognition.auc:.4f} eer={result_a.recognition.eer:.4f} "
        f"n={result_a.num_pairs} skipped={result_a.extra.get('skipped_failed_detections', 0)} "
        f"time={time_a:.1f}s"
    )

    # Baseline B
    det_cfg = {
        "backend": "retinaface",
        "output_size": 112,
        "det_size": [640, 640],
        "model_name": "buffalo_l",
        "weights_path": weights,
        "crop_mode": "bbox_margin",
        "bbox_margin": 0.35,
        "skip_failed": True,
    }
    aligner = create_aligner(det_cfg, device=device)
    assert aligner is not None
    aligner.load(device)
    transform = as_image_transform(aligner)

    t1 = time.perf_counter()
    result_b = _run(pairs, model, matcher, calc, transform, "retinaface", threshold, True)
    time_b = time.perf_counter() - t1
    print(
        f"Baseline B: acc={result_b.recognition.accuracy:.4f} "
        f"auc={result_b.recognition.auc:.4f} eer={result_b.recognition.eer:.4f} "
        f"n={result_b.num_pairs} skipped={result_b.extra.get('skipped_failed_detections', 0)} "
        f"time={time_b:.1f}s"
    )

    d_acc = abs(result_a.recognition.accuracy - result_b.recognition.accuracy)
    d_auc = abs(result_a.recognition.auc - result_b.recognition.auc)
    d_eer = abs(result_a.recognition.eer - result_b.recognition.eer)
    pass_acc = d_acc <= TOL_ACC
    pass_auc = d_auc <= TOL_AUC
    pass_eer = d_eer <= TOL_EER
    ok = pass_acc and pass_auc and pass_eer

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_pairs_requested": args.max_pairs,
        "pairs_loaded": len(pairs),
        "threshold": threshold,
        "device": device,
        "tolerance": {"accuracy": TOL_ACC, "auc": TOL_AUC, "eer": TOL_EER},
        "baseline_a": {
            "accuracy": result_a.recognition.accuracy,
            "auc": result_a.recognition.auc,
            "eer": result_a.recognition.eer,
            "num_pairs": result_a.num_pairs,
            "skipped": result_a.extra.get("skipped_failed_detections", 0),
            "seconds": time_a,
        },
        "baseline_b": {
            "accuracy": result_b.recognition.accuracy,
            "auc": result_b.recognition.auc,
            "eer": result_b.recognition.eer,
            "num_pairs": result_b.num_pairs,
            "skipped": result_b.extra.get("skipped_failed_detections", 0),
            "seconds": time_b,
        },
        "delta_abs": {"accuracy": d_acc, "auc": d_auc, "eer": d_eer},
        "pass": {"accuracy": pass_acc, "auc": pass_auc, "eer": pass_eer, "overall": ok},
        "published_baseline_a_reference": REF_A,
        "notes": (
            "Adapters unchanged: Baseline B feeds shared 112 crops into Buffalo-L "
            "which still runs vendor FaceAnalysis on the crop."
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    json_path = REPORT.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Baseline B tolerance report (Buffalo-L × LFW)",
        "",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Pairs requested: {args.max_pairs}",
        f"- Overall: **{'PASS' if ok else 'FAIL'}**",
        "",
        "| Metric | Baseline A | Baseline B | |Δ| | Tol | Pass |",
        "|--------|-----------:|-----------:|----:|----:|:----:|",
        f"| Accuracy | {result_a.recognition.accuracy:.4f} | {result_b.recognition.accuracy:.4f} | {d_acc:.4f} | {TOL_ACC:.3f} | {pass_acc} |",
        f"| AUC | {result_a.recognition.auc:.4f} | {result_b.recognition.auc:.4f} | {d_auc:.4f} | {TOL_AUC:.3f} | {pass_auc} |",
        f"| EER | {result_a.recognition.eer:.4f} | {result_b.recognition.eer:.4f} | {d_eer:.4f} | {TOL_EER:.3f} | {pass_eer} |",
        "",
        f"JSON: `{json_path.relative_to(ROOT).as_posix()}`",
        "",
        "Published full Baseline A reference (not a subset gate): "
        f"acc={REF_A['accuracy']}, auc={REF_A['auc']}, eer={REF_A['eer']} "
        f"({REF_A['experiment_id']}).",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"OVERALL={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
