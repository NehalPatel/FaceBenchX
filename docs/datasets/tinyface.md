"""TinyFace — FaceBench preparation guide

## Purpose

Low-resolution / tiny face recognition. Research category: **low_resolution**.

## Official source

- TinyFace dataset authors / authorized research releases

FaceBench does **not** download or redistribute TinyFace.

## License / access

Obtain permission as required by the dataset owners. No image binaries in git.

## Expected directory layout

```text
/data/datasets/tinyface/            # <-- dataset.root_path
  Gallery/
    identity_a/
      0001.jpg
  Probe/
    identity_a/
      0001.jpg
  pairs.txt                         # optional
```

If `Gallery/` and `Probe/` are absent, FaceBench falls back to identity folders under the root.

## Protocols

- Primary: identification via `load_gallery()` / `load_probe()`
- Optional verification: `pairs.txt` as `path_a path_b label`
- If no pairs file exists, same/different pairs are synthesized from gallery/probe overlaps

## Integrity checks

Root exists and contains images; Gallery/Probe or identity folders present.

## Minimal YAML

```yaml
dataset:
  name: TinyFace
  root_path: /data/datasets/tinyface
```
