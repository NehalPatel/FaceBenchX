"""LFW (Labeled Faces in the Wild) — FaceBench preparation guide

## Purpose

General unrestricted face verification benchmark (research category: **general**).

## Official source

- Project page: http://vis-www.cs.umass.edu/lfw/
- Common downloads: aligned images archive (`lfw.tgz` / `lfw-deepfunneled.tgz`) and `pairs.txt`

FaceBench does **not** download or redistribute LFW. Obtain the dataset yourself and point YAML `dataset.root_path` at your local copy.

## License / access

LFW is a research dataset released by UMass Amherst. Review the terms on the official site before use. Do not commit image binaries into the FaceBench repository.

## Expected directory layout

`root_path` must point at the **identity folder tree** (the extracted `lfw` directory):

```text
/data/datasets/lfw/                 # <-- dataset.root_path
  Aaron_Eckhart/
    Aaron_Eckhart_0001.jpg
  ...
  pairs.txt                         # preferred location (optional here)
```

`pairs.txt` may also sit **beside** the identity tree:

```text
/data/datasets/
  lfw/                              # <-- dataset.root_path
    Aaron_Eckhart/...
  pairs.txt
```

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.ppm`.

## Pairs protocol file

Standard View-2 `pairs.txt`:

1. First line: fold count (typically `10`).
2. Same-identity lines (3 tokens): `Name idx1 idx2` → `Name/Name_00xx.jpg`
3. Different-identity lines (4 tokens): `NameA idxA NameB idxB`

Dev splits (`pairsDevTrain.txt`, `pairsDevTest.txt`) are also discovered if `pairs.txt` is absent.

Override explicitly in code via `LFWDataset(..., pairs_file="/path/to/pairs.txt")`.

## How FaceBench validates integrity

`LFWDataset.validate_integrity()` checks that:

1. `root_path` exists and is a directory
2. At least one identity subdirectory exists
3. At least one image file exists under an identity folder
4. A pairs protocol file can be resolved

On failure, the result includes `prep_doc: docs/datasets/lfw.md`.

## Minimal YAML snippet

```yaml
experiment:
  name: lfw_smoke
  output_dir: experiments

device: cpu

dataset:
  name: LFW
  root_path: /data/datasets/lfw

model:
  name: buffalo_l   # model adapters arrive in later milestones
```

## Adapter API (Milestone M2)

```python
from facebench.datasets import DatasetFactory

ds = DatasetFactory().create("LFW", root_path="/data/datasets/lfw")
result = ds.validate_integrity()
index = ds.load_dataset()
pairs = ds.load_identity_pairs()
gallery = ds.load_gallery()
probe = ds.load_probe()
```

## Notes

- Primary LFW protocol in FaceBench is **verification** via `load_identity_pairs()`.
- `load_gallery()` / `load_probe()` provide a simple identification split (first image per identity vs remaining) for future scalability experiments.
- Custom / private identity collections are not supported.
