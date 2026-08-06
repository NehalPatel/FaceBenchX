"""CFP-FP (Celebrities in Frontal-Profile) — FaceBench preparation guide

## Purpose

Evaluate recognition under extreme pose variation (frontal vs profile). Research category: **pose**.

## Official source

- Project / paper references for Celebrities in Frontal-Profile (CFP)
- Obtain images and protocol lists from the dataset authors / authorized mirrors

FaceBench does **not** download or redistribute CFP-FP.

## License / access

Follow the original dataset release terms. Do not commit images into this repository.

## Expected directory layout

```text
/data/datasets/cfp/                 # <-- dataset.root_path
  001/
    frontal/
      01.jpg
    profile/
      01.jpg
  002/
    ...
  pairs_fp.txt                      # path_a path_b label
```

Alternate protocol paths searched automatically:

- `Protocol/Pair_list_F.txt`
- `Protocols/Pair_list_F.txt`
- `pairs.txt`

## Pair list format

```text
001/frontal/01.jpg 001/profile/01.jpg 1
001/frontal/01.jpg 002/profile/01.jpg 0
```

Labels: `1`/`true`/`same` = same identity; `0`/`false`/`different` = different.

## Integrity checks

1. Root exists
2. At least one subject subdirectory
3. At least one image under the tree
4. A pair list can be resolved

## Minimal YAML

```yaml
dataset:
  name: CFP-FP
  root_path: /data/datasets/cfp
```
