# Quickstart

- Install (editable):
  - `pip install -e .[dev]`
- Download weights (optional, local cache):
  - List: `gfpgan-download-weights --list`
  - Fetch v1.4: `gfpgan-download-weights -v 1.4`
- Inference on CPU or GPU:
  - `gfpgan-infer --input inputs/whole_imgs --version 1.4 --upscale 2 --device auto`
  - Optional: `--compile default` (Torch 2.x) for speedups
- Helpful flags:
  - `--dry-run` (validate args), `--no-download` (require local weights), `--model-path` (override weights)
  - `--bg_upsampler none` (disable background upsampling), `--bg_precision fp32|fp16|auto`
  - `--detector scrfd|retinaface_*` (switch detector), `--no-parse` (disable parsing)
  - `--manifest outputs.json` to save a simple results manifest
  - `--sweep-weight 0.3,0.5,0.7` to run multiple weights
  - `--print-env` to display versions and CUDA availability
  - Quality: `--jpg-quality 95`, `--png-compress 3`, `--webp-quality 90`
- Colab notebook (UI):
  - https://colab.research.google.com/github/IAmJonoBo/GFPGAN/blob/main/notebooks/GFPGAN_Colab.ipynb

Weights & cache
- Default weights dir: `gfpgan/weights` (override with `GFPGAN_WEIGHTS_DIR`)
- Hugging Face Hub: set `GFPGAN_HF_REPO` to enable hub resolution; set `HF_HUB_OFFLINE=1` for cache-only

See: CLI Usage (./usage/cli.md) and Colab Guide (./usage/colab.md).

Known good versions
- Python 3.11 with Torch 2.x track (e.g., torch 2.4.1, torchvision 0.19.1)
- Basicsr from master when using modern torchvision (CI/Colab default)
