# Manifests & Metrics

This page documents the JSON files written by Restoria and GFPP: `manifest.json`
(run summary) and `metrics.json` (per-image records and optional plan block).

- Both files include a top-level `schema_version` field which is currently
  set to `"2"` for Restoria/GFPP runs.
- `manifest.json` captures args, device, results summary, an optional
  `metrics_file` pointer, and environment info.
- `metrics.json` contains per-image records under `metrics`, plus (in Restoria)
  a `plan` block explaining backend selection and parameters.

## Restoria files

Written by `restoria run`.

### Restoria manifest.json

Shape (fields may be absent if unavailable):

```json
{
  "args": {"backend": "gfpgan", "metrics": "fast", "device": "auto"},
  "device": "cpu",
  "results": [
    {
      "input": "samples/portrait.jpg",
      "backend": "gfpgan",
      "restored_img": "out/portrait.png",
      "metrics": {"arcface_cosine": 0.87}
    }
  ],
  "metrics_file": "metrics.json",
  "env": {
    "runtime": {
      "compile": false,
      "ort_providers": []
    }
  }
}
```

Notes:

- `device` may be `auto`, `cpu`, `cuda`, or `null` if detection failed.
- `env.runtime` contains runtime hints; expanders may add keys in future, but
  existing keys will remain stable.

### Restoria metrics.json

```json
{
  "metrics": [
    {
      "input": "samples/portrait.jpg",
      "backend": "gfpgan",
      "restored_img": "out/portrait.png",
      "metrics": {
        "arcface_cosine": 0.91,
        "lpips_alex": 0.18,
        "dists": 0.14
      }
    }
  ],
  "plan": {
    "backend": "gfpgan",
    "reason": "requested backend",
    "params": {"weight": 0.6}
  }
}
```

Notes:

- When `--metrics off`, values may be empty/omitted; `metrics.json` still
  exists with `{"metrics": [...]}` to keep automation stable.
- `plan` is present in Restoria to make routing decisions auditable.

## GFPP files

Written by `gfpup run`.

### GFPP manifest.json

Built via `src/gfpp/io/manifest.py`:

```json
{
  "args": {"backend": "gfpgan", "metrics": "fast", "device": "auto"},
  "device": "cpu",
  "started_at": 1699999999.12,
  "ended_at": 1699999999.85,
  "results": [
    {"input": "...", "restored_img": "...", "metrics": {"runtime_sec": 0.52}}
  ],
  "models": [],
  "metrics_file": "metrics.json",
  "env": {"torch": "2.x", "cuda": false, "git": "abc123"}
}
```

### GFPP metrics.json

GFPP writes a simple wrapper:

```json
{"metrics": [ {"input": "...", "restored_img": "...", "metrics": {}} ]}
```

Fields in `metrics` may include:

- `arcface_cosine` (higher is better)
- `lpips_alex` (lower is better)
- `dists` (lower is better)
- `runtime_sec`, `vram_mb`, and backend-specific extras

## Stability guarantees

- Key names are stable once released. New keys may be added, but existing keys
  will not be renamed.
- Missing optional metrics must be represented as `null` or omitted; consumers
  should not rely on their presence.
- File locations: both CLIs default to writing into the specified `--output`
  directory. Restoria uses `manifest.json` and `metrics.json` at the root of
  that directory.
