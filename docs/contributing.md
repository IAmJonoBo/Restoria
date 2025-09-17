# Contributing to Docs

This site uses Material for MkDocs and mike for versioned docs.

## Local preview

```bash
pip install -r requirements.txt
pip install mkdocs-material mike
mkdocs serve
```

## Versioned deploy with mike

```bash
mike deploy --push --update-aliases 0.1 latest
```

Notes:

- Keep pages task‑first (Getting Started → Choose Backend → Metrics →
  Hardware).
- Follow WCAG 2.2 AA guidance for accessibility (focus, contrast, keyboard
  navigation, reduced motion).
- Cite standards where relevant: PEP 621, SemVer, Keep a Changelog,
  Contributor Covenant, REUSE/SPDX.
Thanks for considering a contribution! This fork aims to remain close to
upstream while improving practical usage.

- Dev setup
  - `pip install -e .[dev]`
  - Enable pre-commit hooks: `pre-commit install`
  - Lint locally: `ruff check . && black .`
  - Run light tests: `pytest -q tests_light`

- PR guidelines
  - Keep diffs focused and well-described
  - Prefer small, reviewable changes
  - Add or update tests when changing behavior

- Syncing with upstream
  - We periodically pull from upstream `TencentARC/GFPGAN` and resolve conflicts.

Code of Conduct: see `CODE_OF_CONDUCT.md`.
