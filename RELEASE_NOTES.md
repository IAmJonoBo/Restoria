# GFPGAN 2.1.0

Release date: 2025-09-17

Highlights:

- New power CLI JSON modes: `gfpup list-backends --json` and
  `gfpup doctor --json` (schema_version "1").
- Opt-in backends: Guided and Ensemble (safe fallbacks, lazy imports).
- Safer weight loading defaults to `torch.load(weights_only=True)` with
  warned fallback.
- Docs and tests polish; deterministic light test suite; packaging metadata
  cleaned.

Install:

- pip: `pip install gfpgan` (see README for extras like
  `[metrics,arcface,ort]`).

Quick start:

- List backends: `gfpup list-backends --json`.
- Restore (GFPGAN): `gfpup run --input samples/portrait.jpg --backend gfpgan
  --output out/`.
- Auto mode (safe fallback): `gfpup run --auto --input samples/portrait.jpg
  --output out/`.
- Export ONNX: `gfpup export-onnx --output models/onnx/`.

Notes:

- CLI and legacy `inference_gfpgan.py` remain backward compatible.
- Optional metrics/backends degrade gracefully when dependencies or
  weights are missing.

---

For full details, see CHANGELOG.md (section 2.1.0).

