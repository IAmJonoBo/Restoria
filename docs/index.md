# GFPGAN (Fork) — Documentation

<!-- markdownlint-disable MD013 -->

> Note: We are introducing the new Restoria CLI (`restoria`) as the primary
> entry point going forward. Legacy commands remain available during the
> transition. See the migration guide: [guides/migration.md](guides/migration.md).

Welcome to the fork documentation. This fork focuses on modern developer
ergonomics, a smoother Colab experience, and practical inference.

- What’s new vs upstream:
  - Modern CI and linting
  - Colab with interactive UI and compatibility fixes (BasicSR master + modern torchvision)
  - CLI quality-of-life flags and console entrypoint
  - Safer repository defaults and optional light tests

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

Upstream reference: <https://github.com/TencentARC/GFPGAN>

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
