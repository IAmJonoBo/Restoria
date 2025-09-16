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
