## Developer commands

Optional extras and helper CLI are available for local development without changing defaults:

- Install (editable) with extras: `pip install -e ".[dev,metrics,arcface,codeformer,restoreformerpp,ort]"`
- Run single image with explicit backend: `gfpup run --input samples/portrait.jpg --backend gfpgan --metrics full --output out/`
- Try Auto mode (opt-in): `gfpup run --input samples/portrait.jpg --auto --metrics fast --output out/`
- Environment report: `gfpup doctor`

Auto mode is experimental and falls back gracefully when probes or dependencies are missing. Defaults remain unchanged unless opted-in.

# GFPGAN

![GFPGAN Logo](assets/gfpgan_logo.png)

## Professional face restoration powered by generative AI

[![CI](https://github.com/IAmJonoBo/GFPGAN/actions/workflows/ci.yml/badge.svg)](https://github.com/IAmJonoBo/GFPGAN/actions/workflows/ci.yml)
[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/gfpgan)](https://pypi.org/project/gfpgan/)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-brightgreen)](https://IAmJonoBo.github.io/GFPGAN/)
[![Security](https://img.shields.io/badge/security-policy-blue)](SECURITY.md)

**Restore faces in photos with state-of-the-art generative AI.** GFPGAN combines deep learning and generative adversarial networks to intelligently reconstruct facial details from low-quality, damaged, or blurred images.

## Why GFPGAN?

- **Production-ready**: Clean API, robust CLI, batch processing with provenance tracking
- **Multiple backends**: Choose speed vs quality with GFPGAN, CodeFormer, RestoreFormer++
- **Cross-platform**: Windows, macOS, Linux with GPU acceleration (CUDA, MPS, DirectML)
- **Privacy-first**: Process images locally—no cloud uploads required
- **Measurable quality**: Built-in metrics (LPIPS, DISTS, ArcFace) for objective evaluation

## Quick start

Try a single image restoration in 30 seconds:

```bash
# Install
pip install gfpgan

# Restore a photo
gfpgan-infer --input path/to/photo.jpg --version 1.3 --upscale 2

# Results saved to results/ with before/after comparison
```

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IAmJonoBo/GFPGAN/blob/main/notebooks/GFPGAN_Colab.ipynb)

### Quick CLI Usage

- Install (editable): `pip install -e .[dev]`
- Inference: `gfpgan-infer --input inputs/whole_imgs --version 1.3 --upscale 2 --device auto`
- Helpful flags:
  - `--dry-run`: validate args and exit fast
  - `--no-download`: require local weights only
  - `--device {auto,cpu,cuda}`: choose runtime
  - `--bg_upsampler realesrgan|none`: disable background upsample with `none`

Note: on CPU, Real-ESRGAN background upsampling is disabled for speed.

### Colab Features

- Open: [Open in Colab](https://colab.research.google.com/github/IAmJonoBo/GFPGAN/blob/main/notebooks/GFPGAN_Colab.ipynb)
- Features: interactive UI for uploads, URLs, options; preview; ZIP download.
- Compatibility: the notebook installs BasicSR master to match modern torchvision.

### Compatibility notes (Torch/Torchvision/Basicsr)

See `docs/COMPATIBILITY.md` for a quick matrix and notes.

:question: Frequently Asked Questions can be found in [FAQ.md](FAQ.md).

## Core capabilities

### Single image restoration

Restore individual photos with simple CLI or drag-and-drop web interface:

```bash
gfpgan-infer --input damaged_photo.jpg --version 1.3
```

### Batch processing

Process entire folders with consistent settings and quality metrics:

```bash
gfpgan-infer --input photos/ --backend gfpgan --metrics fast --output restored/
```

### Background enhancement

Combine face restoration with background upscaling using Real-ESRGAN:

```bash
gfpgan-infer --input photo.jpg --upscale 2 --bg_upsampler realesrgan
```

## Supported platforms

| Platform | GPU Acceleration | Status |
|----------|------------------|--------|
| **Linux** | CUDA 11.8+ / ROCm | ✅ Full support |
| **Windows** | CUDA / DirectML | ✅ Full support |
| **macOS** | Metal Performance Shaders | ✅ Full support |
| **CPU-only** | All platforms | ✅ Slower, but works |

## Documentation

- **[Getting started guide →](https://IAmJonoBo.github.io/GFPGAN/getting-started/install/)**
- **[API documentation →](https://IAmJonoBo.github.io/GFPGAN/api/)**
- **[Choose the right backend →](https://IAmJonoBo.github.io/GFPGAN/guides/choose-backend/)**

### Key guides

- [Restore a photo](https://IAmJonoBo.github.io/GFPGAN/guides/restore-a-photo/) — CLI and web interface
- [Batch processing](https://IAmJonoBo.github.io/GFPGAN/guides/batch-processing/) — Folders and automation
- [Quality metrics](https://IAmJonoBo.github.io/GFPGAN/guides/metrics/) — Measure restoration quality
- [Hardware optimization](https://IAmJonoBo.github.io/GFPGAN/guides/hardware/) — GPU setup and troubleshooting

:question: **Questions?** Check our [FAQ](https://IAmJonoBo.github.io/GFPGAN/faq/) or [troubleshooting guide](https://IAmJonoBo.github.io/GFPGAN/troubleshooting/).

## Installation

### Quick install

```bash
pip install gfpgan
```

### Development install

```bash
git clone https://github.com/IAmJonoBo/GFPGAN.git
cd GFPGAN
pip install -e ".[dev,metrics,web]"
```

### GPU acceleration

**NVIDIA (CUDA)**: Install PyTorch with CUDA support first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Apple Silicon (MPS)**: Works out of the box with recent PyTorch versions.

**AMD (ROCm)**: Install ROCm-compatible PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
```

For detailed platform-specific instructions, see our [hardware guide](https://IAmJonoBo.github.io/GFPGAN/guides/hardware/).

## CLI usage

### Basic restoration

```bash
# Restore a single image
gfpgan-infer --input photo.jpg --version 1.3

# Process a folder
gfpgan-infer --input photos/ --output results/ --version 1.3

# With background upscaling
gfpgan-infer --input photo.jpg --upscale 2 --bg_upsampler realesrgan
```

### Advanced options

```bash
# Dry run to validate settings
gfpgan-infer --input photos/ --dry-run

# CPU-only mode
gfpgan-infer --input photo.jpg --device cpu

# Disable background upsampling for speed
gfpgan-infer --input photo.jpg --bg_upsampler none
```

### Web interface

Launch the interactive web UI:

```bash
# Basic interface
python -m gfpgan.gradio_app

# API server
uvicorn services.api.main:app --reload
```

## Model comparison

| Model | Speed | Quality | Identity Preservation | Best For |
|-------|-------|---------|---------------------|----------|
| **GFPGAN v1.3** | Medium | High | Good | Natural results (recommended) |
| **GFPGAN v1.2** | Medium | High | Good | Sharp output with beauty makeup |
| **GFPGAN v1** | Medium | Medium | Fair | Basic restoration with colorization |
| **RestoreFormerPlusPlus** | Slow | Highest | Excellent | Professional work, TPAMI 2023 version |
| **RestoreFormer** | Slow | High | Excellent | High-quality restoration |
| **CodeFormer** | Fast | Medium | Good | AI-generated faces, batch processing |
| **CodeFormerColorization** | Fast | Medium | Good | Specialized for colorization tasks |
| **CodeFormerInpainting** | Fast | Medium | Good | Face inpainting and completion |

Choose your model with `--version` flag: `1.3` (default), `1.2`, `1`, `RestoreFormerPlusPlus`, `RestoreFormer`, `CodeFormer`, `CodeFormerColorization`, or `CodeFormerInpainting`.

## Contributing

We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for:

- Setting up the development environment
- Running tests and linting
- Code style and commit conventions
- Submitting pull requests

## Security

Found a security issue? Please report it privately via our [security policy](SECURITY.md).

## License and acknowledgements

GFPGAN is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

### Acknowledgements

This project builds upon the foundational research and implementations from:

- [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) - Original GFPGAN research and implementation (historical reference)
- [xinntao/BasicSR](https://github.com/xinntao/BasicSR) - Super-resolution framework
- [xinntao/facexlib](https://github.com/xinntao/facexlib) - Face detection and analysis utilities

**Note**: This project has been completely unforked from TencentARC/GFPGAN and now operates independently. Models are automatically downloaded from their official sources: [TencentARC/GFPGAN releases](https://github.com/TencentARC/GFPGAN/releases), [sczhou/CodeFormer releases](https://github.com/sczhou/CodeFormer/releases), and [Hugging Face Hub](https://huggingface.co/TencentARC/GFPGANv1) for legacy models.

For a complete list of dependencies and their licenses, see [LICENSES/](LICENSES/).

## Citation

If you use GFPGAN in your research, please cite:

```bibtex
@InProceedings{wang2021gfpgan,
    author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
    title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```

---

**Questions?** Check our [FAQ](https://IAmJonoBo.github.io/GFPGAN/faq/) • [Documentation](https://IAmJonoBo.github.io/GFPGAN/) • [Issues](https://github.com/IAmJonoBo/GFPGAN/issues)
