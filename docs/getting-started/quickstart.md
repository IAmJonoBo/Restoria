# Quick Start

> Note: The new Restoria CLI (`restoria`) is the primary entry point going forward.
> Legacy commands (`gfpgan-infer`, `gfpup`) remain available during the transition.
> See the migration guide: [guides/migration.md](../guides/migration.md).

<!-- markdownlint-disable MD013 -->

- Install (editable):
  - `pip install -e .[dev]`
- Inference on CPU or GPU (recommended):
  - `restoria run --input inputs/whole_imgs --output out/ --backend gfpgan --metrics fast`
  - Legacy shim: `gfpup run --input inputs/whole_imgs --backend gfpgan --metrics fast --output out/`
  - Legacy GFPGAN: `gfpgan-infer --input inputs/whole_imgs --version 1.4 --upscale 2 --device auto`
- Helpful flags:
  - `--dry-run` (validate args), `--no-download` (require local weights), `--model-path` (override weights)
- Colab notebook (UI):
  - [Open in Colab](https://colab.research.google.com/github/IAmJonoBo/Restoria/blob/main/notebooks/Restoria_Colab.ipynb)

See also:

- CLI Usage (./usage/cli.md)
- Colab Guide (./usage/colab.md)

## Listing available backends

Check which backends are available in your environment without heavy downloads:

```bash
gfpup list-backends
# or
restoria list-backends
```

Add `--verbose` for per-backend availability, and `--all` to include experimental ones. For machine-readable output:

```bash
gfpup list-backends --json
# or
restoria list-backends --json
```

The JSON output includes a `schema_version` field (currently "2") and a `backends` dictionary mapping backend name to availability, with an `experimental` flag indicating whether experimental backends are included.

## Environment check (doctor)

```bash
gfpup doctor
# or
restoria doctor
```

For machine-readable output:

```bash
gfpup doctor --json
# or
restoria doctor --json
```

## Optional perceptual metrics

Install optional metrics for identity/perceptual scoring:

```bash
pip install -e .[metrics]
```

Notes:

- Linux: BRISQUE via `imquality[brisque]` is enabled.
- Windows: BRISQUE via `pybrisque` is enabled.
- macOS: BRISQUE is not installed by default to avoid resolver/build issues on newer Python versions; metrics gracefully degrade if BRISQUE is unavailable.
