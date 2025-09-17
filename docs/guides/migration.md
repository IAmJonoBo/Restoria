# Migration to Restoria

This project is being rebranded as Restoria — intelligent image revival. The new
CLI and package are designed to be modular and local‑first while staying
familiar.

## What changes now

- New CLI: `restoria`
- New package import: `import restoria`
- Legacy helper: `gfpup` remains available for a transition period.

## 1:1 command mapping

- `gfpup run --input X --output Y` → `restoria run --input X --output Y`
- `gfpup doctor` → `restoria doctor`

Backends and flags remain the same for common workflows. Advanced options will
continue to evolve behind optional extras and will degrade gracefully if not
installed.

## Installing

- Base (editable): `pip install -e .`
- With metrics: `pip install -e ".[metrics]"`
- With ONNX Runtime: `pip install -e ".[ort]"`

## Notes on compatibility

- The primary distribution metadata still uses the existing project name
  temporarily. The `restoria` console script and package are provided now, and
  the distribution rename will follow a SemVer‑tracked release.
- No hard redirects are required. Shims will be removed in the next major once
  the migration window closes.
