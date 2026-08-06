# Execution plan — LFW × Buffalo-L (first real experiment)

**Status:** Planned — **do not execute** until Phase 10 plan + P0 scaffolding are approved and local data/weights are ready.  
**Config:** [`../configs/lfw_buffalo_l.yaml`](../configs/lfw_buffalo_l.yaml)  
**Prep doc:** [`../../docs/datasets/lfw.md`](../../docs/datasets/lfw.md)

---

## 1. Required downloads

| Asset | Source | Notes |
|-------|--------|-------|
| LFW images | Official LFW / deepfunneled release under dataset terms | Prefer consistently aligned/deepfunneled images for Baseline A |
| `pairs.txt` | Official LFW pairs protocol | Must sit in or beside dataset root as FaceBench expects |
| Buffalo-L pack | InsightFace model zoo (`buffalo_l`) | Place under InsightFace `root` (see `paper/weights/README.md`) |

FaceBench does **not** download or bundle these.

---

## 2. Required software

```bash
pip install -e ".[buffalo,reports,dev]"
# GPU: ensure a compatible onnxruntime-gpu / CUDA stack if device: cuda:0
```

---

## 3. Expected folder structure

```text
/data/datasets/lfw/
  Aaron_Eckhart/
    Aaron_Eckhart_0001.jpg
    ...
  pairs.txt

/data/weights/insightface/
  models/
    buffalo_l/
      *.onnx
      ...
```

Update YAML paths to match your machine (do not commit secrets).

---

## 4. YAML configuration

Locked template: `paper/configs/lfw_buffalo_l.yaml`

- Mode: verification  
- Model: `buffalo_l`  
- Matching: cosine @ 0.40 (also report EER threshold from metrics)  
- Robustness/scalability: off for Baseline A  

Debug first:

```yaml
evaluation:
  max_pairs: 100
```

---

## 5. Expected runtime

| Stage | GPU (approx.) | CPU (approx.) |
|-------|---------------|---------------|
| Debug `max_pairs: 100` | minutes | tens of minutes |
| Full ~6000 pairs | ~15–45 min | multi-hour |

Warm-up and first-time ONNX/InsightFace pack prepare add overhead.

---

## 6. Commands (when authorized to execute)

```bash
facebench validate-config --config paper/configs/lfw_buffalo_l.yaml
facebench env --pretty

# Debug (recommended first)
facebench run --config paper/configs/lfw_buffalo_l.yaml
# with max_pairs set in YAML

# Full run: remove max_pairs, ensure NO --allow-stub
facebench run --config paper/configs/lfw_buffalo_l.yaml
```

---

## 7. Expected outputs

```text
experiments/<experiment_id>/
  config.snapshot.yaml
  env.json
  metrics/summary.json
  runs/LFW__buffalo_l/
    reports/report.html
    reports/report.md
    metrics/*.csv|json
    figures/roc_curve.png
    figures/confusion_matrix.png
```

After P1: also `manifest.json`.

Curate into `paper/results/lfw_buffalo_l_<id>/`.

---

## 8. Possible failures

| Symptom | Cause | Action |
|---------|-------|--------|
| Integrity failed | Wrong LFW root / missing pairs | Fix layout per `docs/datasets/lfw.md` |
| Import / BuffaloLBackendError | Missing `[buffalo]` deps | Reinstall extras |
| No faces detected | Bad images / wrong LFW variant | Spot-check; try deepfunneled |
| CUDA / ORT errors | GPU runtime mismatch | Set `device: cpu` |
| Accidental stub metrics | `--allow-stub` | Abort; not a paper run |

---

## 9. Debugging checklist

- [ ] Config validates
- [ ] LFW integrity OK
- [ ] One-image embedding works (shape ~512, finite)
- [ ] Debug `max_pairs` run completes
- [ ] HTML + MD + CSV + JSON present
- [ ] ROC + confusion figures present
- [ ] `env.json` records GPU/CPU as expected
- [ ] Acceptance checklist (`acceptance_checklist.md`) all green
- [ ] Results curated under `paper/results/`

---

## 10. Research note (Baseline A)

This first run uses **InsightFace Buffalo-L’s bundled detector** (vendor pack).  
A later **Baseline B** will use shared RetinaFace crops for all models (see Phase 10 plan §5). Report Baseline A clearly in methods.
