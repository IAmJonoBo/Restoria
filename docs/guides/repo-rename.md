# Repository rename & migration checklist

This page lists safe, incremental steps to rename the repository and
PyPI package while keeping users productive.

Goal: migrate branding and distribution names with zero breaking changes
for existing workflows until users opt in.

## Principles

- Backward compatibility by default: existing CLIs and flags continue to work.
- Shims only: legacy entry points remain but display a one-time deprecation notice.
- Deterministic behavior: same inputs produce same plan/output under the same version.
- Clear docs and changelog: announce deprecation windows and timelines.

## Step-by-step

1. Repository rename

- Rename GitHub repo to the new name in Settings.
- Verify redirects are auto-created (GitHub handles this). Keep old
  remote URLs in docs for one minor release.
- Update badges and links in README and docs.

1. Package metadata

- Update `pyproject.toml` project name and URLs.
- Keep console scripts for both new and legacy entry points during the
  transition.
- Bump minor version and add a changelog entry noting the rename and
  deprecation window.

1. CI and docs

- Update release workflows to publish to the new package name (if
  changing on PyPI).
- Keep a compatibility wheel that provides legacy entry points if
  feasible.
- Use mike to version the docs. Add a banner on latest docs highlighting
  the new name.

1. Deprecation policy

- Show a one-time deprecation notice in legacy CLI recommending the new
  CLI name and commands.
- Provide a migration guide (see Migration to Restoria) with
  side-by-side examples.
- After two minor versions, consider removing the legacy shim, following
  SemVer (major) if behavior changes.

1. Weight and model registries

- Keep weight resolution modules stable; do not rename on disk abruptly.
- Provide redirects or lookup fallbacks for old registry names to new
  names.

## Validation

- Run light tests (`tests_light`) locally and in CI. Avoid heavy model downloads.
- Smoke-test CLI paths: plan-only, dry-run, and normal run for a small sample image.
- Confirm manifests and metrics are still written with stable key names.

## Rollback

- If users report blockers, you can temporarily restore older wheel
  versions and mark the latest as yanked on PyPI while issuing a patch
  release.

## See also

- Migration to Restoria (User Guides)
- Governance → Versioning policy
