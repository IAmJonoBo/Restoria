# Fork Notes

This fork aims to keep GFPGAN easy to install, test, and integrate while staying close to upstream behavior.

## Goals
- Modern packaging with `uv` for fast, reproducible environments.
- Default to the PyTorch 2.x track; keep compatibility options for older setups.
- Continuous Integration: lint (`ruff`, `black`) and tests (`pytest`) on Python 3.10/3.11.
- Safer repository defaults (branch protection on `main`).
- Maintain API surface and outputs as close to upstream as practical.

## Non‑goals
- Feature divergence for its own sake.
- Maintaining long‑term patches that deviate from upstream without clear value.

## Coordination with Upstream
- Track upstream `TencentARC/GFPGAN` for bug fixes and improvements.
- Prefer small, reviewable PRs; rebase/merge from upstream periodically.

## Quick Links
- Fork repository: https://github.com/IAmJonoBo/GFPGAN
- Upstream repository: https://github.com/TencentARC/GFPGAN
