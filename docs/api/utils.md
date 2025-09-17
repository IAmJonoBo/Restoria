# API: Utilities

General helpers and public types exposed by the modular layer.

## Public types

```python
from gfpp.restorers.base import RestoreResult

# Fields
# input_path: str | None
# restored_path: str | None
# restored_image: any | None
# cropped_faces: list[str]
# restored_faces: list[str]
# metrics: dict[str, any]
```

## Manifest helpers

```python
from gfpp.io.manifest import RunManifest, write_manifest

man = RunManifest(args={"backend": "gfpgan", "metrics": "fast"}, device="cpu")
man.results.append({
  "input": "inputs/a.jpg",
  "restored_img": "out/a.png",
  "metrics": {"arcface_cosine": 0.87}
})
write_manifest("out/manifest.json", man)
```

Notes:

- `RunManifest.env` includes runtime info such as git SHA and torch version
  when available.
- Manifests and metrics are JSON files intended to be stable and machine-readable.

## Image IO helpers

The project favors lazy imports for heavy dependencies (cv2, PIL, torch) and
permits multiple image array formats as `restored_image` in `RestoreResult`.
When building higher-level pipelines, normalize to numpy arrays in BGR or RGB
as appropriate for your stack.

## Logging and progress

The CLI emits structured logs and writes run manifests. For library usage,
prefer passing a `progress` callback in your integration code and capture
timings around `prepare` and `restore` calls.
