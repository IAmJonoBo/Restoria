# Contributing (Fork)

Thanks for considering a contribution! This fork aims to remain close to upstream while improving practical usage.

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
