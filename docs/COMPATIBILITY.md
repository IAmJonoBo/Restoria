# Compatibility Matrix (Fork)

This fork targets a smooth developer and Colab experience with modern Torch/Torchvision while remaining compatible with upstream behavior where practical.

- Python 3.10
  - Preferred: Torch 1.x track (`-E torch1`)
  - Example pins: `torch==1.13.1`, `torchvision==0.14.1`
- Python 3.11
  - Preferred: Torch 2.x track (`-E torch2`)
  - Example range: `torch>=2.2,<3`, `torchvision>=0.17,<0.20`
- Basicsr
  - Colab and CI install Basicsr from `master` to match modern `torchvision` functional API.
  - On older Torch/Torchvision (1.x), the PyPI Basicsr release is typically fine.

Notes
- Colab installs: Torch defaults to CPU wheels (GPU optional via CUDA 12.1 index); Basicsr is installed from `master` for compatibility.
- Real-ESRGAN background upsampling is disabled on CPU in this fork for performance—use GPU runtime in Colab for best results.
- If you need fully deterministic runs, use `--seed` and prefer CPU (or set CUDA deterministic/env flags).
