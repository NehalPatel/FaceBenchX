"""YouTube Faces (YTF) — FaceBench preparation guide

## Purpose

Video-based face verification with motion blur, pose change, and compression. Research category: **video**.

## Official source

- YouTube Faces DB project pages / authorized mirrors

FaceBench does **not** download or redistribute YTF.

## License / access

Comply with YTF terms. Frame images must remain outside the FaceBench git tree.

## Expected directory layout

```text
/data/datasets/ytf/                 # <-- dataset.root_path
  Person_Name/
    1/
      0.jpg
      1.jpg
    2/
      0.jpg
  pairs.txt                         # path_or_video_a path_or_video_b label
```

Also accepted:

```text
/data/datasets/ytf/
  frame_images_DB/
    Person_Name/1/*.jpg
```

## Protocols

- `pairs.txt` / `splits.txt` with `path_a path_b label`
- Video directories referenced in pair lists are reduced to the **first frame** in Milestone M3 (`frame_aggregation: first`)
- If no pairs file exists, pairs are synthesized across frames/videos

## Integrity checks

Root (or `frame_images_DB/`) contains person folders and frame images.

## Minimal YAML

```yaml
dataset:
  name: YTF
  root_path: /data/datasets/ytf
```

## Note

All models in a given experiment must share the same frame aggregation choice. Additional aggregations (mean pooling of embeddings) arrive with the evaluation pipeline in later milestones.
