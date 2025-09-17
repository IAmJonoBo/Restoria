# Restoria — Face Restoration Toolkit

<!-- markdownlint-disable MD013 -->

> Restoria is the project brand and primary CLI (`restoria`). GFPGAN remains the
> default backend and legacy commands remain available for compatibility
> (`gfpup`, `gfpgan-infer`). See the migration guide:
> [guides/migration.md](guides/migration.md).

Welcome to Restoria’s documentation. Restoria is a modular, CLI‑first face
restoration toolkit with multiple interchangeable backends (GFPGAN by default),
deterministic planning, and machine‑readable outputs.

- Highlights
  - Modular backends: GFPGAN (default), CodeFormer, RestoreFormer++
  - Deterministic runs (`--seed`, `--deterministic`) and stable manifests/metrics
  - Modern CI and link-checked docs; optional metrics with graceful fallback
  - Legacy compatibility preserved: existing GFPGAN workflows continue to work

Quick links

- Getting started
  - Quickstart: getting-started/quickstart.md
  - Troubleshooting: troubleshooting.md
- Reference
  - CLI Usage: usage/cli.md
  - Backend Matrix: BACKEND_MATRIX.md
  - Compatibility: COMPATIBILITY.md
- Contribute
  - Contributing: governance/contributing.md

Upstream research reference (historical): <https://github.com/TencentARC/GFPGAN>

## Quick tiles

- :material-rocket: **Quickstart** — Start here to install and
  restore your first image. [:material-arrow-right: Open quickstart]
  (getting-started/quickstart.md)

- :material-console: **CLI usage** — Learn the `restoria` and `gfpup`
  commands, flags, and outputs. [:material-arrow-right: Explore CLI]
  (usage/cli.md)

- :material-book-open-variant: **User guides** — Deep dives: choose a
  backend, metrics, performance, and troubleshooting.
  [:material-arrow-right: Browse guides] (guides/choose-backend.md)

- :material-api: **API** — Run the FastAPI server or integrate via
  Python APIs. [:material-arrow-right: REST API] (usage/api.md)
