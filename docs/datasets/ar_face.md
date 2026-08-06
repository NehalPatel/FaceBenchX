"""AR Face Database — FaceBench preparation guide

## Purpose

Occlusion (sunglasses, scarves) and illumination robustness. Research category: **occlusion**.

## Official source

- AR Face Database (Ohio State / Aleix Martinez). Access typically requires a request form.

FaceBench does **not** download or redistribute AR Face images.

## License / access

Users must obtain AR Face independently and comply with its license. Prep docs only describe expected layout.

## Expected directory layout

```text
/data/datasets/ar_face/             # <-- dataset.root_path
  m-001/
    *.bmp
  w-001/
    *.bmp
  pairs.txt                         # optional
```

## Protocols

- Optional `pairs.txt` (`path_a path_b label`)
- Otherwise same/different pairs are synthesized from multi-image identities

Filename tokens such as `sunglass` / `scarf` are recorded as occlusion cues in sample metadata.

## Integrity checks

Root + identity subdirectories + at least one image.

## Minimal YAML

```yaml
dataset:
  name: AR-Face
  root_path: /data/datasets/ar_face
```
