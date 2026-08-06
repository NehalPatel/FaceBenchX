"""CPLFW (Cross-Pose LFW) — FaceBench preparation guide

## Purpose

Cross-pose / expression LFW variant, harder than standard LFW. Research category: **pose**.

## Official source

- CPLFW project pages / mirrors used in ArcFace-style evaluation toolkits

FaceBench does **not** download or redistribute CPLFW.

## License / access

Respect the dataset authors' terms. Do not bundle images in git.

## Expected directory layout

```text
/data/datasets/cplfw/               # <-- dataset.root_path
  Name/
    Name_0001.jpg
  pairs_CPLFW.txt
```

Also accepts LFW-style `pairs.txt` beside or inside the root.

## Pair list formats

1. LFW-style 3/4-token lines (with optional numeric header)
2. Path/label lines: `rel/path_a.jpg rel/path_b.jpg 1`

## Integrity checks

Root + identity folders + images + resolvable pairs file.

## Minimal YAML

```yaml
dataset:
  name: CPLFW
  root_path: /data/datasets/cplfw
```
