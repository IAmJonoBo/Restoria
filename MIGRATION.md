# Migration Guide

This fork upgrades the CLI, Colab, and CI. Key notes:

- Python/Torch: Default target is Python 3.11 with Torch 2.x; Python 3.10 is supported via constraints.
- Basicsr: Use upstream master to match modern torchvision APIs.
- CLI: New optional flags (`--workers`, `--auto`, `--auto-hw`, `--select-by`, `--backend`). Backwards compatibility is preserved; defaults match previous behavior.
- Colab: GPU-aware Torch install; optional Autopilot toggles; skips heavy installs in CI via `NB_CI_SMOKE`.
- CI: Adds nbmake for notebook smoke, model link checks, and docs deployment.

Breaking considerations
- If you pin older torchvision/torch versions, ensure Basicsr is compatible (recommend master).
- `--backend codeformer` is a placeholder and requires additional setup; default remains GFPGAN.

Reproducibility
- Use `uv` to manage pinned environments; see `scripts/setup_uv.sh` and `constraints-*` files.

