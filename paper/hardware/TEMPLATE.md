# Hardware profile template

Fill one file per machine used for paper numbers (copy this file).

```yaml
profile_id: workstation_01
owner: <name>
recorded_at: YYYY-MM-DD
hostname: <hostname>

cpu:
  model: <e.g. AMD Ryzen 9 / Intel Xeon>
  cores_logical: <n>

memory_gb: <n>

gpu:
  present: true
  model: <e.g. NVIDIA RTX 4090>
  memory_gb: <n>
  driver_version: <x.y>

cuda:
  version: <e.g. 12.1>
  cudnn_version: <optional>

os:
  name: <Windows 11 / Ubuntu 22.04>
  version: <build or release>

python:
  version: "3.11.x"
  venv: <path or description>

notes: |
  Any known nondeterminism, thermal throttling, or shared-host caveats.
```

After a run, attach the matching `experiments/<id>/env.json` path under `notes`.
