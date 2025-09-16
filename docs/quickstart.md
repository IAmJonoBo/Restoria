# Quickstart

- Install (editable):
  - `pip install -e .[dev]`
- Inference on CPU or GPU:
  - `gfpgan-infer --input inputs/whole_imgs --version 1.4 --upscale 2 --device auto`
- Helpful flags:
  - `--dry-run` (validate args), `--no-download` (require local weights), `--model-path` (override weights)
- Colab notebook (UI):
  - https://colab.research.google.com/github/IAmJonoBo/GFPGAN/blob/main/notebooks/GFPGAN_Colab.ipynb

See: CLI Usage (./usage/cli.md) and Colab Guide (./usage/colab.md).


## Listing available backends

Check which backends are available in your environment without heavy downloads:

```bash
gfpup list-backends
```

Add `--verbose` for per-backend availability, and `--all` to include experimental ones. For machine-readable output:

```bash
gfpup list-backends --json
```

The JSON output includes a `schema_version` field (currently "1") for forward compatibility, plus `experimental` and a `backends` dictionary mapping backend name to a boolean availability.

## Environment check (doctor)

Print environment and availability information:

```bash
gfpup doctor
```

For machine-readable output:

```bash
gfpup doctor --json
```

## Optional perceptual metrics

You can install optional metrics for identity/perceptual scoring:

```bash
pip install -e .[metrics]
```

Notes:
- Linux: BRISQUE via `imquality[brisque]` is enabled.
- Windows: BRISQUE via `pybrisque` is enabled.
- macOS: BRISQUE is not installed by default to avoid resolver/build issues on newer Python versions; metrics gracefully degrade if BRISQUE is unavailable.
