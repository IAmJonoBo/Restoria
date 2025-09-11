# Changelog

All notable changes to this fork are documented here.

This project uses a simple, human-written changelog inspired by Keep a Changelog. Dates use ISO‑8601 (YYYY‑MM‑DD).

## [Unreleased]
- Changed: Modernize packaging with `uv`; default to PyTorch 2.x track; add constraints for Python 3.10/3.11.
- Added: CI workflow (`.github/workflows/ci.yml`) with Python 3.10/3.11 matrix, `ruff` and `black` checks, and pytest.
- Changed: Update `.gitignore` to ignore local virtualenv folders.
- Removed: Stale `.github/workflows/no-response.yml` workflow.
- Repo: Enabled branch protection on `main` (require 1 review, no force pushes, linear history, conversation resolution).

## [v1.3.8-fork1] - 2025-09-11
- Initial fork tag from upstream `v1.3.8`.
- Docs: README fork branding, CI badge, and clone URL updates.

---

Links
- Upstream: https://github.com/TencentARC/GFPGAN
- Fork: https://github.com/IAmJonoBo/GFPGAN
