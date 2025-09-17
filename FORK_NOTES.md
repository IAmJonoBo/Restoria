# Project Independence Notes

This project was originally forked from TencentARC/GFPGAN but has now been
completely unforked to operate as an independent project.

## Goals

- Modern packaging with `uv` for fast, reproducible environments.
- Default to the PyTorch 2.x track; keep compatibility options for older setups.
- Continuous Integration: lint (`ruff`, `black`) and tests (`pytest`) on Python 3.10/3.11.
- Safer repository defaults (branch protection on `main`).
- Maintain API surface and outputs for compatibility.

## Independence Status

- **Complete separation**: This project no longer tracks or syncs with TencentARC/GFPGAN
- **Model hosting**: All models now served from Hugging Face Hub (TencentARC/GFPGANv1)
- **Independent development**: Features and improvements developed independently

## Historical Context

- Originally forked from: <https://github.com/TencentARC/GFPGAN>
- Original paper: [GFPGAN: Towards Real-World Blind Face Restoration with
  Generative Facial Prior](https://arxiv.org/abs/2101.04061)
- All original research credit remains with TencentARC team

## Quick Links

- Project repository: <https://github.com/IAmJonoBo/Restoria>
- Model repository: <https://huggingface.co/TencentARC/GFPGANv1>
