# FaceBench Phase 10 — Scientific Validation Plan

**Document type:** Review + validation roadmap (architecture frozen)  
**Status:** Awaiting approval before further framework changes beyond the single P0 scaffolding task  
**Date:** 2026-08-06  
**Framework version:** `0.1.0`  
**Related:** [FaceBench-Design.md](FaceBench-Design.md), [chatgpt-conversion.md](chatgpt-conversion.md), [dataset-suggestion.md](dataset-suggestion.md)

---

## Executive recommendation (read first)

FaceBench is a **stable, stub-validated benchmarking library**. It is **not yet publication-validated**.

**Highest-priority next action (P0):** Create the locked `paper/` experiment suite scaffolding (configs, hardware, reproduction, checklists) so real LFW × Buffalo-L work has a reproducible home. **Do not add models/datasets or redesign the core.**

**Immediately after P0 approval:** Implement experiment `manifest.json` generation (P1), then run the first real LFW × Buffalo-L acceptance experiment (P2 execution), with shared RetinaFace alignment deferred until after the first clean baseline is measured (P3).

**Publication readiness score: 58 / 100** (framework complete; scientific lock-in and real-run evidence missing).

---

## 1. Current project health assessment

### What is healthy

| Area | Assessment |
|------|------------|
| Package layout | Matches design §9; installable via `pip install -e ".[dev]"` |
| Config / experiments | YAML normalize, `datasets: all`, experiment IDs, `env.json`, history |
| Dataset adapters | All 8 public adapters + prep docs + synthetic/contract tests |
| Model adapters | All 5 adapters with factory, extras, stub backends |
| Metrics / matcher | Recognition + compute metrics; cosine/euclidean |
| Reports / figures | HTML/MD/CSV/JSON + matplotlib figure suite |
| CLI / orchestrator | End-to-end `facebench run` with multi dataset×model matrix |
| Evaluation axes | Verification, identification, synthetic robustness, scalability |
| Lint / unit tests | Ruff/Black clean; ~87 unit/contract tests (stub/synthetic only) |
| License | MIT present |

### What is weak / incomplete

| Area | Assessment |
|------|------------|
| Real-data validation | **None locked** — no paper results, no acceptance runs |
| Detect/align | **Passthrough only**; **not wired** into orchestrator/verification |
| Reproducibility manifest | Partial (`env.json` + config snapshot); **no** `manifest.json` with checksums/git/weights |
| Paper suite | **Missing** until P0 scaffolding |
| CI | No `.github/workflows` observed |
| Docs site / CONTRIBUTING / CHANGELOG | Missing |
| AdaFace / MagFace | Require user-supplied checkpoints; **no bundled architecture factory** for official nets |
| YTF | Frame aggregation = `first` only (not mean-pool as design suggests) |
| Dual registries | `core.registry` and `datasets.category_registry` bridge — works but easy to drift |

### Health verdict

**Framework maturity: high.**  
**Scientific maturity: low–medium.**  
Safe to freeze architecture and shift to validation, not feature expansion.

---

## 2. Objective 1 — Architecture / code review findings

### Architecture inconsistencies

1. **Shared detect/align is documented but unused.** `facebench.detection.PassthroughAligner` exists; orchestrator and `run_verification` call `model.generate_embedding(path)` directly. Models that detect internally (Buffalo-L, Dlib) therefore run a **different** preprocess path than FaceNet/AdaFace/MagFace (resize-only).
2. **Design §16.1** requires identical detect→align→embed for all models; current code violates that for fair comparison claims.
3. **Buffalo-L uses InsightFace’s bundled detector** inside `FaceAnalysis.get()`, while FaceNet assumes a face-centric crop already present — confounding recognition with detection quality.
4. **Identification gallery construction** for most datasets is “first image per identity / rest as probe” (`gallery_first_probe_rest`), not always the dataset’s official ID protocol (TinyFace is the main exception with Gallery/Probe dirs).
5. **M7 “category charts”** and paper table automation are not first-class; aggregation exists but is not manuscript-locked.

### Duplicate / overlapping code

- Near-identical recognizer scaffolding across FaceNet / AdaFace / MagFace / Buffalo-L / Dlib (`allow_stub`, backend resolve, load logging).
- Pair/gallery helpers centralized in `datasets/common.py` (good); some adapters still reimplement small variants.
- Category registry duplicated conceptually between `core/registry.py` and `datasets/category_registry.py` (bridged, but two entry points).

### Unfinished / placeholder / stub implementations

| Item | Status |
|------|--------|
| `DeterministicStubBackend` | Intentional for CI; must never appear in paper runs |
| `--allow-stub` CLI | Smoke only |
| `PassthroughAligner` | Placeholder for shared align |
| AdaFace/MagFace without official net builder | Needs user `model=` or full pickled module |
| YTF `frame_aggregation` | Only `"first"`; mean pooling deferred |
| Detection package | Not integrated into eval loop |
| Paper suite / manifests / acceptance tests | Missing (this phase) |

### TODOs in code

No widespread `TODO`/`FIXME` markers. Incomplete behavior is expressed via stubs, “M3 supports first only”, and docstrings pointing to future aligners.

### Weak abstractions

- No formal `Aligner` protocol consumed by orchestrator.
- No `WeightSpec` (path + checksum + version) type for paper provenance.
- No dataset **version/hash** capture (only `root_path`).
- Report `extra` bag used for robustness/scalability — works, but paper exporters should treat these as first-class result types.

### Missing documentation

- No experiment reproduction guide under `paper/reproduction/`
- No hardware lock file template
- No model weight acquisition runbook beyond extras install strings
- No CONTRIBUTING / API docs site / CHANGELOG
- Design doc footer still says “awaiting M1 approval” (stale status line)

### Missing tests (scientific, not unit)

- No real LFW integrity acceptance test
- No real weight load smoke
- No golden metric ranges vs published LFW numbers
- No paper acceptance checklist automation
- No bit-reproducibility / seed regression on real backend

---

## 3. Objective 2 — Model adapter review

### Summary table

| Model | Official basis | Input | Dim | L2-norm | GPU | Weight story | Paper risk |
|-------|----------------|-------|-----|---------|-----|--------------|------------|
| FaceNet | `facenet-pytorch` InceptionResnetV1 (VGGFace2) | 160×160 RGB, [-1,1] | 512 | Yes | Torch CUDA | Auto-download or local `.pt` | Medium — no MTCNN align in adapter |
| Dlib | dlib ResNet v1 + shape predictor | Full image, HOG detect + landmarks | 128 | Yes | CPU only | Local `.dat` files required | Medium — own detector |
| Buffalo-L | InsightFace `buffalo_l` pack | BGR → internal RetinaFace + ArcFace | ~512 | Yes | ONNX CUDA/CPU | InsightFace `root` pack | **Lowest** for first real run |
| AdaFace | User torch checkpoint | 112×112 ArcFace norm | 512 | Yes | Torch CUDA | **Must** supply weights (+ maybe `model=`) | High — architecture binding fragile |
| MagFace | User torch checkpoint | 112×112 ArcFace norm | 512 | Yes | Torch CUDA | Same as AdaFace | High — same |

### Suggested improvements (do not implement yet unless prioritized)

1. **Document official weight URLs + SHA256** in `paper/weights/README.md` and per-model notes.
2. **Disable stub path** when `evaluation.paper_mode: true` (hard fail if stub used).
3. **FaceNet:** optional shared aligner input; pin `facenet-pytorch` / torchvision versions for paper.
4. **Dlib:** document that landmark align is model-internal; either accept as “bundled-align exception” or feed pre-aligned crops only.
5. **Buffalo-L:** for fair shared-align mode, prefer recognition-only ONNX (`w600k_r50`) on pre-aligned 112 crops once RetinaFace pipeline exists; keep current FaceAnalysis path as “vendor pack” baseline.
6. **AdaFace/MagFace:** ship thin official architecture constructors (IR-50/IR-100) that load `state_dict` from known filenames; record checkpoint hash in manifest.
7. **Similarity:** all adapters L2-normalize → cosine is correct; euclidean on normalized vectors is equivalent to angular distance — document threshold incompatibility across dims (128 vs 512).

---

## 4. Objective 3 — Dataset adapter review

### Summary table

| Dataset | Pairs protocol | Gallery/probe | Integrity | Notes / inconsistencies |
|---------|----------------|---------------|-----------|-------------------------|
| LFW | Official-style `pairs.txt` folds | First / rest heuristic | Strong | Best-validated; gallery ≠ official unrestricted protocol |
| CFP-FP | `path_a path_b label` (+ Protocol alts) | Heuristic | OK | Depends on user-normalized pair list vs original CFP scripts |
| CPLFW | Similar path/label pairs | Heuristic | OK | Layout sensitivity |
| AgeDB-30 | AgeDB-style pairs | Heuristic | OK | Filename identity parsing has flat fallback |
| TinyFace | Optional pairs; ID-oriented | **Native Gallery/Probe** | OK | Best ID protocol alignment |
| AR Face | Path/label or identity dirs | Heuristic | OK | Occlusion metadata not richly modeled |
| ChokePoint | Pairs / splits flexible | Heuristic | OK | Surveillance layouts vary widely |
| YTF | Video/dir pairs | Heuristic | OK | **`first` frame only** — weak vs video SOTA practice |

### Cross-cutting inconsistencies

1. Verification is primary and well-supported; identification is mostly **derived**, not official CMC protocols (except TinyFace splits).
2. Integrity checks verify presence of images/protocols, **not** dataset version hashes or pair-count expectations (e.g. LFW 6000 pairs).
3. Prep docs exist for all 8; none yet have a “paper frozen layout” checksum section.
4. No adapter records `dataset_version` / license acknowledgment into run metadata.

### Suggested improvements (later)

- Add optional `expected_pair_count` / `expected_identity_count` to integrity for paper configs.
- YTF: add `mean` frame-embedding aggregation (orchestrator-level) before paper video claims.
- Freeze per-dataset layout fingerprints in `paper/configs/datasets/*.md`.

---

## 5. Objective 4 — Detection / alignment evaluation & migration plan

### Current state

- `PassthroughAligner` loads RGB only.
- **Not called** by orchestrator.
- Buffalo-L and Dlib detect internally; others resize whole image.

### Should RetinaFace-based shared alignment replace passthrough?

**Yes, for publication claims of a unified protocol** (design FR-05 / §16.1).

**No, not before the first Buffalo-L × LFW baseline** if the goal is to validate the pipeline end-to-end quickly — Buffalo-L’s InsightFace pack already includes RetinaFace and is the fastest path to a real number.

### Recommendation

| Stage | Align strategy | Purpose |
|-------|----------------|---------|
| Baseline A (first paper smoke) | Buffalo-L vendor detect+embed (current) | Prove orchestrator + LFW + reports |
| Baseline B (fair comparison) | Shared RetinaFace → 112×112 aligned RGB → all models | Main manuscript protocol |
| Exception column | Dlib / vendor-align noted | Transparency |

### Migration plan (implement later — not now)

1. Define `BaseAligner` protocol: `align(image) -> AlignedFace` with landmarks + bbox metadata.
2. Implement `RetinaFaceAligner` (InsightFace SCRFD/RetinaFace or `insightface` model zoo), optional extra `[align]`.
3. Wire orchestrator: load image → align once → pass ndarray crop to `generate_embedding`.
4. Add model flag `accepts_aligned_crop: bool`; Buffalo-L recognition-only backend for aligned mode.
5. Config:
   ```yaml
   detection:
     backend: retinaface   # or passthrough | insightface_scrfd
     output_size: 112
   ```
6. Contract tests with synthetic faces; acceptance test: same crop tensor hash across models’ preprocess inputs.
7. Document deviation table in paper methods.

**Effort estimate:** 3–5 days engineering + 1 day validation.

---

## 6. Objective 5 — Paper Experiment Suite structure

Target tree (created by P0 task):

```text
paper/
├── README.md
├── configs/           # locked experiment YAMLs
├── hardware/          # machine profiles
├── weights/           # acquisition notes + checksum ledger (no binaries in git)
├── results/           # copied / linked curated run outputs (gitignored bulk)
├── tables/            # manuscript tables (CSV/MD)
├── figures/           # manuscript figures
├── logs/              # curated logs
└── reproduction/      # step-by-step repro guides + acceptance checklists
```

Rules:

- No dataset images or weight binaries in git.
- `paper/weights/` holds README + `checksums.sha256` templates only.
- `paper/results/` gitignored except tiny placeholders / README.

---

## 7. Objective 6 — Experiment manifest design

Every paper run should write `manifest.json` beside `env.json`:

```json
{
  "experiment_id": "...",
  "timestamp": "ISO-8601",
  "git_commit": "...",
  "framework_version": "0.1.0",
  "dataset": {"name": "LFW", "root_path": "...", "version": "...", "protocol_file": "...", "pair_count": 6000},
  "model": {"name": "buffalo_l", "weights_path": "...", "weight_version": "...", "weight_sha256": "..."},
  "matching": {"method": "cosine", "threshold": 0.40},
  "detection": {"backend": "insightface_buffalo_pack", "shared_align": false},
  "seed": 42,
  "environment": {
    "os": "...", "cpu": "...", "gpu": "...", "cuda_version": "...",
    "python_version": "...", "packages": {}
  },
  "execution_time_s": 0.0,
  "outputs": {"reports": [], "figures": [], "metrics": []}
}
```

**Gap vs today:** `env.json` covers much of environment; missing git commit, weight checksum, dataset version, execution time, structured dataset/model blocks, and a single paper-facing file.

**Implementation task ID:** P1 (after P0).

---

## 8. Objective 7 — Paper Acceptance Tests (design)

Not unit tests — gate real runs. Proposed location: `paper/reproduction/acceptance_checklist.md` + later `tests/acceptance/` (manual/opt-in).

| ID | Check | Pass criterion |
|----|-------|----------------|
| A01 | Dataset validated | `validate_integrity().ok`; expected pair count |
| A02 | Model loaded | `load_model` without stub; backend class name recorded |
| A03 | Embeddings generated | Finite vectors; expected dim |
| A04 | Recognition completed | Scores/labels length match pairs |
| A05 | Metrics calculated | AUC/EER/F1 present in JSON |
| A06–A09 | Reports | HTML, MD, CSV, JSON exist |
| A10–A11 | Figures | ROC + confusion matrix PNGs |
| A12 | Hardware recorded | `env.json` non-empty GPU/CPU fields as applicable |
| A13 | Manifest generated | `manifest.json` schema valid |
| A14 | No stub | Manifest `backend` ≠ DeterministicStub |

---

## 9. Objective 8 — First real experiment plan (LFW × Buffalo-L)

**Do not execute in this document’s approval cycle until P0 scaffolding is reviewed.**

### Goal

Produce one complete real verification run proving FaceBench can generate publication artifacts.

### Required downloads

1. **LFW** aligned or deepfunneled images + official `pairs.txt` (user obtains under LFW terms).
2. **InsightFace buffalo_l** pack (auto-fetched by InsightFace on first run *or* pre-placed under a local models root).
3. Python env: `pip install -e ".[buffalo,reports,dev]"`.

### Suggested local layout

```text
/data/datasets/lfw/           # or paper-local path outside git
  <identity>/*.jpg
  pairs.txt
/data/weights/insightface/    # InsightFace root containing models/buffalo_l/
```

### YAML (locked under `paper/configs/`)

See `paper/configs/lfw_buffalo_l.yaml` (P0).

### Expected runtime (order-of-magnitude)

| Hardware | Full LFW (~6k pairs, 2 embeds each) |
|----------|--------------------------------------|
| CUDA GPU (modern) | ~15–45 minutes |
| CPU only | several hours |

Use `evaluation.max_pairs: 100` for a **debug** run first (~minutes).

### Expected outputs

```text
experiments/<experiment_id>/
  config.snapshot.yaml
  env.json
  manifest.json          # after P1
  metrics/summary.json
  runs/LFW__buffalo_l/
    metrics/...
    reports/report.html|md
    figures/roc_curve.png
    figures/confusion_matrix.png
```

### Possible failures

| Failure | Likely cause | Debug |
|---------|--------------|-------|
| Integrity failed | Wrong root / missing pairs.txt | Follow `docs/datasets/lfw.md` |
| `BuffaloLBackendError` | Missing insightface/onnxruntime | Reinstall `[buffalo]` |
| No faces detected | Unaligned junk images / grayscale issues | Spot-check images; try deepfunneled LFW |
| CUDA errors | onnxruntime-gpu mismatch | Fall back `device: cpu` |
| Stub used accidentally | `--allow-stub` | Forbidden for paper |

### Debugging checklist

1. `facebench validate-config --config paper/configs/lfw_buffalo_l.yaml`
2. Integrity-only Python snippet on LFW root
3. Load Buffalo-L on one image, print embedding shape
4. `facebench run ...` with `max_pairs: 20`
5. Confirm reports/figures
6. Full run without max_pairs

---

## 10. Objective 9 — Publication Readiness Checklist

| Item | Status (as of plan) |
|------|---------------------|
| □ Framework architecture frozen | **Proposed yes** — freeze now |
| □ Dataset adapters validated (synthetic) | ✓ |
| □ Dataset adapters validated (real LFW) | ✗ |
| □ Recognition adapters validated (stub) | ✓ |
| □ Recognition adapters validated (real weights) | ✗ |
| □ Detection/alignment finalized | ✗ (passthrough / vendor) |
| □ Metrics verified (unit) | ✓ |
| □ Metrics verified vs published baselines | ✗ |
| □ Reports verified (unit) | ✓ |
| □ Reports verified (real run) | ✗ |
| □ Figures verified (real run) | ✗ |
| □ Experiment reproducibility verified | ✗ |
| □ Hardware documented | ✗ |
| □ Environment reproducibility verified | Partial (`env.json`) |
| □ First benchmark completed (LFW×Buffalo-L) | ✗ |
| □ Five-model benchmark completed | ✗ |
| □ Multi-dataset benchmark completed | ✗ |
| □ Manuscript tables generated | ✗ |
| □ Supplementary material prepared | ✗ |
| □ Manifest.json on every paper run | ✗ |
| □ Paper acceptance tests green | ✗ |

Live checklist copy: `paper/reproduction/publication_readiness_checklist.md` (P0).

---

## 11. Remaining technical risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Unfair model comparison without shared align | High | RetinaFace migration (P3); document Baseline A vs B |
| AdaFace/MagFace checkpoint load failures | High | Pin official IR architectures + checksums |
| InsightFace auto-download non-determinism | Medium | Vendor pack into local `weights/` + SHA256 |
| YTF overclaim with first-frame only | Medium | Defer video RQ until mean aggregation |
| Threshold not comparable across models | Medium | Report EER threshold + ROC; avoid single global threshold claims |
| GPU nondeterminism | Medium | Record seed + note nondeterminism; CPU verification subset |
| Accidental stub in paper run | High | `paper_mode` hard fail (P1/P2) |

## 12. Remaining research risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Results not matching literature LFW numbers | High | Align protocol (pairs, aligned LFW, detect); discuss deltas honestly |
| Synthetic robustness over-interpreted as RQ3 | Medium | Keep category datasets primary (design §16.5) |
| Scalability N limited by dataset size | Low | Skip + record reason (already implemented) |
| Scope creep into new models/datasets | High | Architecture freeze rule |

---

## 13. Publication readiness score (0–100)

| Dimension | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Architecture completeness | 15 | 90 | 13.5 |
| Dataset coverage (code) | 10 | 85 | 8.5 |
| Model coverage (code) | 10 | 75 | 7.5 |
| Metrics/reports | 10 | 90 | 9.0 |
| Protocol fairness (align) | 15 | 35 | 5.25 |
| Real-run evidence | 20 | 5 | 1.0 |
| Reproducibility (manifest/paper suite) | 15 | 25 | 3.75 |
| Docs/CI/release polish | 5 | 40 | 2.0 |
| **Total** | 100 | — | **≈ 58** |

---

## 14. Priority-ranked implementation tasks

| ID | Task | Effort | Depends | Implement now? |
|----|------|--------|---------|----------------|
| **P0** | Create `paper/` suite scaffolding + LFW×Buffalo-L plan artifacts + checklists | 0.5 day | — | **YES (this cycle)** |
| **P1** | Implement `manifest.json` writer (git, checksums, seed, timing) | 1 day | P0 | After approval |
| **P2** | Paper acceptance harness + first LFW×Buffalo-L **execution** (user machine) | 1–2 days | P0, P1 | After approval |
| **P3** | Shared RetinaFace aligner + orchestrator wiring | 3–5 days | P2 baseline | After Baseline A |
| **P4** | Five-model LFW comparison under shared align | 2–3 days | P3 | Later |
| **P5** | Multi-dataset (pose/age) locked configs + tables | 3–5 days | P4 | Later |
| **P6** | AdaFace/MagFace official architecture loaders + weight ledger | 2 days | P4 | Parallel possible |
| **P7** | YTF mean aggregation + video note | 1–2 days | P5 | Later |
| **P8** | CI workflow (stub-only) + CONTRIBUTING + CHANGELOG | 1 day | — | Parallel polish |
| **P9** | Manuscript table/figure export scripts from `paper/results` | 1–2 days | P4–P5 | Later |

---

## 15. Validation roadmap & timeline

```text
Week 1:  P0 scaffolding (done this cycle) → approve plan
         P1 manifest.json
         P2 debug max_pairs run → full LFW×Buffalo-L
Week 2:  Analyze Baseline A vs literature; write methods note
         Start P3 RetinaFace shared align
Week 3:  P3 complete; P4 five-model LFW
Week 4:  P5 multi-dataset subset (CFP-FP, CPLFW, AgeDB-30)
Week 5:  P6 weight hardening; P9 tables; draft supplementary
Week 6:  P7/P8 polish; freeze v0.2.0-paper-rc
```

---

## 16. Clear recommendation — what to implement first

**Implement P0 only in this cycle:**

1. Create `paper/` directory structure.
2. Add locked `paper/configs/lfw_buffalo_l.yaml` (paths as placeholders).
3. Add reproduction guide, acceptance checklist, publication checklist, hardware template, weights README.
4. Stop for human review.

**Do not** implement RetinaFace, new models/datasets, or major refactors until P0+plan are approved and the first real Buffalo-L run is scheduled.

---

## Appendix A — Mapping to design milestones

| Design | Phase 10 stance |
|--------|-----------------|
| M8 optional axes | Code done; scientific use secondary |
| M9 paper experiments | **This phase’s focus** |
| M10 open-source polish | Parallel low priority (P8) |

---

**End of Phase 10 Scientific Validation Plan**

*Next gate: approve this plan and P0 scaffolding, then authorize P1 (`manifest.json`).*
