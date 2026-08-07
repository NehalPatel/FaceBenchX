# Baseline B — Shared RetinaFace Migration Plan

**Status:** Implementation in progress (this document)  
**Constraint:** Recognition adapters (`facebench/models/*`) are **not** modified in this phase.  
**Reference baseline:** LFW × Buffalo-L Baseline A — experiment `20260805T203632Z_paper_lfw_buffalo_l_f2e9f94c` (accuracy 0.9817, AUC 0.9892, EER 0.0238).

---

## 1. Goals

1. Add a **shared** detect → align → 112×112 RGB crop pipeline (InsightFace SCRFD / RetinaFace-family) usable by all models.
2. Wire it into FaceBench via config + orchestrator **without** changing recognizer adapters.
3. Keep **Baseline A** as the default path (no shared align) so prior Buffalo-L results remain reproducible.
4. Validate that Buffalo-L under Baseline B stays within an **acceptable tolerance** of Baseline A on the same LFW pairs.

---

## 2. Non-goals (deferred)

| Item | Why deferred |
|------|----------------|
| Recognition-only Buffalo-L / Dlib backends | Requires adapter changes (`accepts_aligned_crop`) |
| Five-model LFW under Baseline B | Depends on this pipeline + weight readiness |
| Native `manifest.json` in core orchestrator | Separate P1 track |
| Replacing vendor detect inside adapters | Phase after Baseline B lands |

---

## 3. Architecture (additive)

```text
YAML detection:
  backend: none | passthrough | retinaface

none / omitted     → Baseline A (path → model.generate_embedding)  [default]
passthrough        → load RGB only; optional transform hook tests
retinaface         → Baseline B (path → shared align → ndarray crop → embed)
```

```text
facebench/detection/
  align.py          # AlignedFace, PassthroughAligner, BaseAligner protocol
  retinaface.py     # RetinaFaceAligner (InsightFace SCRFD pack)
  factory.py        # create_aligner(config)
```

Orchestrator builds an optional `image_transform` and passes it to `run_verification` / robustness (existing hook). Identification gets the same transform for gallery/probe embeds.

**Important:** Until adapters are updated, Buffalo-L still runs its **internal** FaceAnalysis on the crop (double detect). That is intentional for this phase and is documented as Baseline B (shared crop) + vendor embed.

---

## 4. Implementation steps

| Step | Deliverable | Done when |
|------|-------------|-----------|
| A | This migration plan | Merged under `docs/` |
| B | `RetinaFaceAligner` + factory + `FaceDetectionError` | Unit tests pass with stub/mock |
| C | Config `detection:` normalization | `load_config` accepts Baseline A/B YAMLs |
| D | Orchestrator / identification wiring | `image_transform` applied when `backend: retinaface` |
| E | YAML templates + local Baseline B config | Paper configs present |
| F | Tolerance verification vs Baseline A | Scripted A vs B on shared pair subset; report under `paper/results/` |

---

## 5. YAML schema

```yaml
detection:
  backend: retinaface          # none | passthrough | retinaface
  output_size: 112             # ArcFace-family crop size
  det_size: [640, 640]
  model_name: buffalo_l        # InsightFace pack providing SCRFD
  weights_path: null           # optional InsightFace root (models/...)
  skip_failed: true            # skip pairs when shared detect finds no face
```

Default when section omitted: `backend: none` (Baseline A).

---

## 6. Tolerance policy (Buffalo-L)

Compare **Baseline A** vs **Baseline B** on the **same** LFW pairs (same threshold 0.40, cosine, CPU):

| Metric | Accept if |
|--------|-----------|
| Accuracy | \(\lvert \Delta \rvert \le 0.015\) |
| AUC | \(\lvert \Delta \rvert \le 0.015\) |
| EER | \(\lvert \Delta \rvert \le 0.020\) |

Primary verification uses a locked subset (`max_pairs`, e.g. 500) for runtime. Full 6000-pair Baseline B is a follow-up once adapters support recognition-only mode.

Absolute drift vs the published full Baseline A numbers is recorded but **not** required to match exactly on a subset.

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Double detection changes scores | Tolerance band; later recognition-only backend |
| Extra CPU cost | Cache shared crops in paper runners if needed |
| No face on shared detect | `skip_failed` + log skipped pairs |
| `insightface` API variance | Try `allowed_modules=['detection']`; fall back to full pack |

---

## 8. Rollout checklist

- [ ] Default configs unchanged (Baseline A)
- [ ] `detection.backend: retinaface` enables Baseline B
- [ ] No edits under `facebench/models/**` recognizers
- [ ] Unit tests for factory / aligner contract
- [ ] A vs B tolerance script results archived
- [ ] Stop — do not start five-model comparison in this task

---

## 9. Next phase (after stop)

1. Add `accepts_aligned_crop` + recognition-only Buffalo-L/Dlib paths.  
2. Five-model LFW under Baseline B + McNemar.  
3. Multi-dataset RQ3 suite.
