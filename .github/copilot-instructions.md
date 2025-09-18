## Copilot Project Instructions – GFPGAN Toolkit

Purpose: Enable AI agents to contribute productively without breaking existing user workflows. This repo is an unforked, production-focused face restoration toolkit with multiple interchangeable backends and optional metrics/performance paths.

### Core Architecture
1. Legacy user CLI: `inference_gfpgan.py` (`gfpgan-infer` entry point) – MUST remain backward compatible (flags, defaults, output layout: `results/`, `restored_imgs/`, etc.).
2. New modular layer (under `src/gfpp/`):
   - `restorers/` adapters (e.g. `gfpgan.py`, `codeformer.py`, `restoreformerpp.py`) implement a lightweight `Restorer` protocol (`base.py`).
   - `engines/` provide execution variants (eager, `torch.compile`, ONNX Runtime) with safe fallbacks.
   - `metrics/` contains optional metric wrappers (ArcFace, LPIPS, DISTS) – every metric object exposes `available()` and must degrade silently if deps/weights absent.
   - `io/` handles manifests, loading, saving (`RunManifest` writing includes args, runtime metadata, per-image metrics).
   - `cli.py` supplies the new power CLI (`gfpup`) with subcommands: `run`, `doctor`, `export-onnx`.
3. Weight resolution: `gfpgan/weights.py` handles local vs HF vs URL download with sha256; never duplicate weight logic in new code – call `resolve_model_weight`.
4. UI: `gfpgan/gradio_app.py` (kept simple, lazy imports). Future UI enhancements must not force new hard dependencies into the base install.

### Non‑Breaking Guarantees (Enforce in PRs)
- Do NOT rename, remove, or change semantics of existing public flags (`gfpgan-infer` or `gfpup run`). Add new flags as optional, default-off.
- Default backend remains GFPGAN; auto/orchestrated selection only when `--auto` / `--auto-backend` is explicitly set.
- Experimental backends (DiffBIR, HYPIR) or performance paths (ORT, compile) must gracefully fallback and emit warnings instead of raising.

### Typical Developer Workflows
- Editable install with extras:
  `pip install -e ".[dev,metrics,arcface,codeformer,restoreformerpp,ort]"`
- Run tests: `python -m pytest tests/` (CI uses `scripts/test.sh` which wraps `uv run pytest`). Keep tests light: skip heavy model downloads unless explicitly marked.
- New CLI usage examples (ensure they continue to work when adding code):
  - `gfpup run --input samples/portrait.jpg --backend gfpgan --metrics full --output out/`
  - `gfpup run --input samples/portrait.jpg --auto --metrics fast --output out/`
  - `gfpup doctor` (environment + available backends/providers)

### Patterns & Conventions
- Lazy imports: heavy libs (torch, cv2, basicsr) are imported inside functions/methods (`prepare`, metrics methods) to reduce CLI startup latency and keep dry-run fast.
- Metrics & optional features: Always wrap import/setup in try/except; expose an `available()` method; return `None` for missing results rather than raising.
- Manifest: Any new runtime info should be added under `man.env["runtime"]` keys or per-image `rec["metrics"]`. Keep keys snake_case and stable.
- Determinism: When adding randomness, respect `--seed` & `--deterministic` (see `_set_deterministic` in `cli.py`).
- Weight selection heuristics or planning logic must be deterministic for identical inputs (seed + image) and produce an explanation string (used in plan/manifest).
- Avoid embedding large constants/URLs in multiple places; centralize model specs in the registry / weights logic.

### Adding a New Backend (Minimal Checklist)
1. Create `src/gfpp/restorers/<name>.py` implementing `prepare(cfg)` + `restore(image, cfg) -> RestoreResult`.
2. Lazy import external packages inside `prepare`.
3. Decide optional extra group in `pyproject.toml` (keep empty list if no extra deps yet).
4. Update backend enumeration in `cli.py` choices (both `--backend` and any auto/orchestrator mapping).
5. Add availability check (try import) before use; fallback to gfpgan with a warning.
6. Write a light unit test that mocks image array and asserts `RestoreResult.restored_image` is not None (or falls back safely).

### Auto / Orchestrator (Current State)
- Basic heuristic path wired via `--auto`/`--auto-backend`. If advanced orchestrator module exists (`gfpp.core.orchestrator.plan`), it returns a Plan with: backend, params, postproc, reason. Always ensure failures silently disable auto mode (fallback to user-chosen backend).

### Performance Paths
- `--compile` triggers optional `torch.compile` via `engines/torch_compile.py`; wrap in try/except and leave model unmodified on failure.
- ONNX Runtime backend (`gfpgan-ort`) loads pre-exported model; provider selection logged under manifest runtime (key: `ort_providers`). Never hard-fail on missing ORT, just warn.

### Metrics Strategy
- Identity: ArcFace primary (higher cosine better). If unavailable, skip — do not compute approximate substitutes.
- Perceptual: LPIPS + DISTS only in `--metrics full` mode; fast mode includes ArcFace only.
- Store numeric values under `metrics.json` and per-image `metrics` field; do not change existing key names retroactively.

### Error Handling & Fallback Philosophy
- User-visible restoration should proceed even if: metrics missing, compile fails, ORT unavailable, experimental backend import errors.
- Log with `[WARN]` prefix and continue.

### Testing Guidance
- Unit tests must not download multi-hundred-MB weights. Mock or short-circuit heavy code paths.
- Add new tests under `tests/` mirroring existing naming patterns; keep them parallel-friendly and deterministic.

### Quick Reference (Do / Avoid)
DO: lazy import torch/cv2, return `None` metric values, keep CLI flags stable, write manifest entries.
AVOID: global side effects at import time, raising on optional feature absence, duplicating weight resolution logic, changing output directory structure.

---
If adding major functionality, append a short section here summarizing new extension points. Keep this file concise (< ~120 lines) and actionable.

### New: Precision/Tiling Flags Parity
- Both CLIs (gfpup and restoria run) accept `--precision`, `--tile`, and `--tile-overlap`.
- Defaults are safe and off by default; features should degrade gracefully when unsupported.

### New: Plugin Entry Points for Backends
- External backends can register via `entry_points` group `gfpp.restorers` mapping `name -> module:Class`.
- The registry merges built-ins with discovered plugins; plugin errors are isolated and must not break listing or selection.
