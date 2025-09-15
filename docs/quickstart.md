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
