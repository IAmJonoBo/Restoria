# Release Playbook

This checklist keeps releases predictable and auditable.  Every step should be executed in order.

## 1. Prepare the branch

- Ensure `main` is green: `make lint`, `make test`, `nox -s tests_full`.
- Update dependencies if required and regenerate `requirements-locked.txt`.
- Refresh SBOM and compare against the current main branch: `uv run --with cyclonedx-bom cyclonedx-py -e -o sbom.json` then `git diff -- sbom.json`.

## 2. Update artefacts

- Bump `VERSION` and add a changelog entry in `CHANGELOG.md`.
- Regenerate documentation if content changed: `make docs-build`.
- Run the benchmark harness on the canonical dataset and commit the CSV/HTML output (`bench/out/`).
- For notebooks, execute the smoke test: `make nb-smoke`.

## 3. Regenerate release metadata

- Run `restoria doctor --json > doctor-report.json` and attach to the release PR.
- Generate an SBOM diff (CI now uploads `sbom.diff`); review for unexpected additions.
- Create or update `scripts/cache_weights.py` output if new weights were added.

## 4. Final PR checks

- Ensure `docs/governance/contributing.md` and `docs/governance/release-playbook.md` links remain valid.
- Tag the release PR with `release` and request review from maintainers listed in `CODEOWNERS`.
- When approved, squash merge and tag the commit (e.g. `git tag v1.4.1 && git push origin v1.4.1`).

## 5. Publish

- Build distributions: `python -m build`.
- Upload to PyPI with `twine upload dist/*`.
- Publish the GitHub release, attaching:
  - Generated SBOM (`sbom.json`) and diff (`sbom.diff`)
  - Benchmark CSV/HTML outputs
  - Doctor report and any relevant artefacts (e.g. Colab notebook exports)

## 6. Post-release

- Regenerate documentation site: `mkdocs gh-deploy` or trigger the docs workflow.
- Update the HuggingFace Space / Colab notebook badges to reference the new tag if required.
- Announce the release in the project channels and archive the release checklist for future reference.
