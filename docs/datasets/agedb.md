"""AgeDB-30 — FaceBench preparation guide

## Purpose

Age-invariant face verification across large age gaps. Research category: **age**.

## Official source

- AgeDB dataset release / mirrors used in face recognition benchmarks

FaceBench does **not** download or redistribute AgeDB.

## License / access

Follow AgeDB release terms. Do not commit images to the repository.

## Expected directory layouts

### Flat files (classic AgeDB naming)

```text
/data/datasets/agedb/               # <-- dataset.root_path
  0_MariaCallas_35_f.jpg
  1_MariaCallas_70_f.jpg
  agedb_30_pairs.txt
```

### Identity folders

```text
/data/datasets/agedb/
  MariaCallas/
    MariaCallas_0001.jpg
  agedb_30_pairs.txt
```

## Pair list formats

- `agedb_30_pairs.txt` / `pairs.txt`
- LFW-style tokens or `path_a path_b label`

## Integrity checks

Root exists, at least one image is present, pairs file resolves.

## Minimal YAML

```yaml
dataset:
  name: AgeDB-30
  root_path: /data/datasets/agedb
```
