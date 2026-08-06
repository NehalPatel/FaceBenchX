# FaceBench

**FaceBench** is a unified benchmarking framework for comparative evaluation of modern deep face recognition models.

> Status: **Manuscript draft** — LFW × Buffalo-L Baseline A paper draft available at [paper/manuscript/FaceBench-LFW-BuffaloL-Draft.md](paper/manuscript/FaceBench-LFW-BuffaloL-Draft.md).

## Requirements

- Python 3.10+

## Installation

```bash
pip install -e ".[dev]"
# Optional extras: [reports], [facenet], [dlib], [buffalo], [adaface], [magface]
```

## CLI

```bash
facebench --help
facebench list-datasets
facebench list-models
facebench validate-config --config configs/examples/smoke_m1.yaml
facebench env --pretty

# Dry-run (metadata only)
facebench run --config configs/examples/cli_smoke.yaml --dry-run

# Smoke evaluation with stub backends (no heavy model deps)
facebench run --config configs/examples/cli_smoke.yaml --allow-stub --max-pairs 8 --no-figures

# Production run (real weights + local dataset paths in YAML)
facebench run --config configs/examples/smoke_m1.yaml

facebench history --output-root experiments --limit 10
facebench report --experiment-dir experiments/<experiment_id>
```

### Benchmark execution (Phase 9)

YAML controls protocol mode and optional axes:

```yaml
evaluation:
  mode: verification   # or identification

robustness:
  enabled: true
  transforms: [blur, gaussian_noise, jpeg, low_illumination]

scalability:
  enabled: true
  identity_counts: [10, 100, 500, 1000, 5000]
```

Examples: `configs/examples/robustness_lfw.yaml`, `configs/examples/scalability_lfw.yaml`.

```python
from facebench.evaluation import (
    run_verification,
    run_identification,
    run_robustness_suite,
    run_scalability_ladder,
)
```

### LFW adapter (M2)

```python
from facebench.datasets import DatasetFactory

ds = DatasetFactory().create("LFW", root_path="/data/datasets/lfw")
assert ds.validate_integrity().ok
index = ds.load_dataset()
pairs = ds.load_identity_pairs()
```

Preparation guide: [docs/datasets/lfw.md](docs/datasets/lfw.md).

### Recognizers (M4/M5)

```python
from facebench.models import ModelFactory, DeterministicStubBackend

# Production extras:
# pip install 'facebench[facenet]' | '[dlib]' | '[buffalo]' | '[adaface]' | '[magface]'
facenet = ModelFactory().create("facenet", device="cpu")
buffalo = ModelFactory().create("buffalo_l", device="cpu", weights_path="/data/insightface")
adaface = ModelFactory().create("adaface", device="cpu", weights_path="/data/adaface.pt")
magface = ModelFactory().create("magface", device="cpu", weights_path="/data/magface.pt")

# Tests / smoke without heavy deps:
stub = DeterministicStubBackend(embedding_dim=512)
model = ModelFactory().create("buffalo_l", backend=stub)
model.load_model()
embedding = model.generate_embedding(image_rgb_ndarray)
```

### Metrics (Phase 6)

```python
from facebench.metrics import MetricCalculator, ComputeProfiler
from facebench.matcher import create_matcher

matcher = create_matcher("cosine")
scores = [matcher.score(e1, e2) for e1, e2 in embedding_pairs]
labels = [1, 0, 1, ...]  # same/different

calc = MetricCalculator()
recog = calc.recognition(labels, scores, threshold=0.4)
print(recog.auc, recog.eer, recog.f1)

profiler = calc.create_profiler(warmup=2)
profiler.track_embedding(lambda: model.generate_embedding(image))
comp = calc.computational(profiler)
print(comp.avg_embedding_time_s, comp.throughput_fps)
```

### Reports (Phase 7)

```bash
pip install -e ".[reports]"   # matplotlib (+ optional seaborn/Jinja2)
```

```python
from facebench.reports import ReportGenerator

gen = ReportGenerator("experiments/demo_run")
data = gen.write_per_dataset(
    experiment_id="demo_run",
    experiment_name="lfw_buffalo",
    dataset_name="LFW",
    model_name="buffalo_l",
    recognition=recog,
    computational=comp,
    y_true=labels,
    y_score=scores,
)
gen.write_aggregated([data])  # comparison tables + charts
```

## Development

```bash
ruff check .
black --check .
pytest
```

## Design & validation

- Architecture: [docs/FaceBench-Design.md](docs/FaceBench-Design.md)
- Phase 10 scientific validation plan: [docs/Phase10-Scientific-Validation-Plan.md](docs/Phase10-Scientific-Validation-Plan.md)
- Paper experiment suite: [paper/README.md](paper/README.md)

## License

MIT — see [LICENSE](LICENSE).
