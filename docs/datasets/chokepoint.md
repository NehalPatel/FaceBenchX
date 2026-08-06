"""ChokePoint Dataset — FaceBench preparation guide

## Purpose

Surveillance / multi-camera recognition with walking subjects. Research category: **surveillance**.

## Official source

- ChokePoint dataset (NICTA / Data61). Access usually requires registration.

FaceBench does **not** download or redistribute ChokePoint.

## License / access

Request the dataset from the official maintainers. Do not bundle media files in git.

## Expected directory layout

```text
/data/datasets/chokepoint/          # <-- dataset.root_path
  subject_001/
    cam_a/
      *.jpg
  subject_002/
    ...
  pairs.txt                         # optional
  gallery.txt                       # optional path list
  probe.txt                         # optional path list
```

Identity may be a subject folder or a session/camera folder depending on how you organize extracted frames. FaceBench indexes immediate child directories as identities and collects images recursively.

## Protocols

- Optional path/label `pairs.txt`
- Optional `gallery.txt` / `probe.txt` path lists
- Otherwise gallery/probe and pairs are derived from the identity tree

## Integrity checks

Root + subdirectories + at least one image.

## Minimal YAML

```yaml
dataset:
  name: ChokePoint
  root_path: /data/datasets/chokepoint
```
