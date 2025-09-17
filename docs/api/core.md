# API: Core

Core modules behind the modular gfpp library and CLI.

- Orchestrator and Plan (deterministic planning)
- Registry and Restorer protocol (pluggable backends)
- Manifest IO (run manifests and metrics)

## Orchestrator

The orchestrator computes a deterministic Plan for a given input and options.
It aims to be pure and repeatable for identical inputs and the same seed.

Plan dataclass fields:

- backend: str
- params: dict[str, any]
- postproc: dict[str, any]
- reason: str
- confidence: float (0..1)
- quality: dict[str, float | None]
- faces: dict[str, any]
- detail: dict[str, any]

Example:

```python
from gfpp.core import orchestrator

plan = orchestrator.plan(
    "samples/portrait.jpg",
    {"backend": "gfpgan", "compile": False, "ort_providers": []},
)
print(plan.backend, plan.params, plan.reason)
```

## Registry and Restorer protocol

Backends are resolved via a registry (`gfpp.core.registry`). Each backend implements:

- prepare(cfg: dict) -> None (lazy import heavy deps)
- restore(image: np.ndarray, cfg: dict) -> RestoreResult

The result contains at least:

- restored_image: np.ndarray | None
- metrics: dict (optional, can be empty)

## Manifest IO

Runs typically write two files into the output directory by default:

- metrics.json
- manifest.json

Example manifest shape:

```json
{
  "args": {"backend": "gfpgan", "metrics": "fast", "device": "auto"},
  "device": "cpu",
  "results": [
    {
      "input": "inputs/a.jpg",
      "restored_img": "out/a.png",
      "metrics": {"arcface_cosine": 0.87}
    }
  ],
  "metrics_file": "metrics.json",
  "env": {
    "runtime": {"compile": false, "ort_providers": []},
    "git": "abc123",
    "torch": "2.x",
    "cuda": false
  }
}
```

Notes:

- Optional features degrade gracefully; missing metrics return `None`.
- Heavy dependencies (torch, cv2, metrics) are imported lazily inside
  functions and guarded with try/except.
