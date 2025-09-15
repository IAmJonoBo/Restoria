# GFPGAN

## Professional face restoration powered by generative AI

Restore---

Ready to get started? [**Install GFPGAN →**](getting-started/install.md)h state-of-the-art generative AI. GFPGAN combines deep learning and generative adversarial networks to intelligently reconstruct facial details from low-quality, damaged, or blurred images.

---

## Three ways to get started

### :material-image: Restore a photo

Single image restoration with CLI or web interface

=== "CLI"
    ```bash
    pip install gfpgan
    gfpgan-infer --input photo.jpg --version 1.4
    ```

=== "Web UI"
    ```bash
    python -m gfpgan.gradio_app
    # Open http://localhost:7860
    ```

[**→ Restore a photo guide**](guides/restore-a-photo.md){ .md-button .md-button--primary }

### :material-folder-multiple: Batch process

Process entire folders with consistent quality

```bash
gfpgan-infer --input photos/ --backend gfpgan --metrics fast --output results/
```

[**→ Batch processing guide**](guides/batch-processing.md){ .md-button .md-button--primary }

### :material-chart-line: Measure quality

Objective evaluation with LPIPS, DISTS, and ArcFace metrics

```bash
gfpgan-infer --input photos/ --metrics detailed --report-path quality_report.json
```

[**→ Quality metrics guide**](guides/metrics.md){ .md-button .md-button--primary }

---

## Core capabilities

| Feature | Description |
|---------|-------------|
| **Multiple backends** | GFPGAN, CodeFormer, RestoreFormer++ - choose speed vs quality |
| **Cross-platform** | Windows, macOS, Linux with GPU acceleration (CUDA, MPS, DirectML) |
| **Privacy-first** | Process images locally—no cloud uploads required |
| **Production-ready** | Clean API, robust CLI, batch processing with provenance tracking |
| **Measurable quality** | Built-in metrics for objective evaluation |

## Quick links

- [**API documentation**](api/index.md) — FastAPI auto-docs and examples
- [**Choose the right backend**](guides/choose-backend.md) — Speed vs quality comparison
- [**Hardware setup**](HARDWARE_GUIDE.md) — GPU optimization and troubleshooting
- [**FAQ**](faq.md) — Common questions and solutions

## Platform support

| Platform | GPU Acceleration | Status |
|----------|------------------|--------|
| **Linux** | CUDA 11.8+ / ROCm | ✅ Full support |
| **Windows** | CUDA / DirectML | ✅ Full support |
| **macOS** | Metal Performance Shaders | ✅ Full support |
| **CPU-only** | All platforms | ✅ Slower, but works |

---

Ready to get started? [**Install GFPGAN →**](getting-started/installation.md)

---

Ready to get started? [**Install GFPGAN →**](getting-started/install.md)Documentation

Welcome to the fork documentation. This fork focuses on modern developer ergonomics, a smoother Colab experience, and practical inference.

- What’s new vs upstream:
  - Modern CI and linting
  - Colab with interactive UI and compatibility fixes (BasicSR master + modern torchvision)
  - CLI quality-of-life flags and console entrypoint
  - Safer repository defaults and optional light tests

Quick links
- Quickstart: ./quickstart.md
- CLI Usage: ./usage/cli.md
- Colab Guide: ./usage/colab.md
- Compatibility Matrix: ./COMPATIBILITY.md
- Contributing: ./contributing.md

Historical reference: https://github.com/TencentARC/GFPGAN (original research)
Model repository: https://huggingface.co/TencentARC/GFPGANv1
