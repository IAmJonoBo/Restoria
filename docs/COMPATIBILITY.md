<<<<<<< HEAD
# Compatibility Matrix

This fork supports two primary tracks to balance stability and modern stacks.

- Torch 1.x (stable, Python 3.10): use `extras = [dev, torch1]` or `-c constraints-3.10-compat.txt`.
- Torch 2.x (modern, Python 3.11+): use `extras = [dev, torch2]`. Install BasicSR from master if needed.

Known-good combinations

- Python 3.10 + Torch 1.13.1 + torchvision 0.14.1 + basicsr 1.4.2
- Python 3.11 + Torch 2.4.1 + torchvision 0.19.1 + basicsr (master)

Matrix (guide)
- CPU (3.11): Torch 2.4.1, Torchvision 0.19.1, Basicsr master, facexlib >=0.3.0
- CUDA 12.1 (3.11): install torch/torchvision/torchaudio from PyTorch CUDA 12.1 index; Basicsr master
- OpenCV: use prebuilt wheels (>=4.8) for 3.11; older tracks pin to <4.8

Notes
- Basicsr 1.4.2 uses `torchvision.transforms.functional_tensor` which was removed in torchvision 0.15+. For Torch 2.x stacks, install BasicSR from GitHub master.
- CPU runs disable Real-ESRGAN background upsampling by default for speed.
- Apple Silicon: prefer Python 3.10 + constraints for prebuilt wheels; Torch 2.x on arm64 is improving but may pull heavier builds.

Quick recipes
- 3.10 stable (Torch 1.x): `scripts/setup_uv.sh --python 3.10 --track torch1`
- 3.11 modern (Torch 2.x): `scripts/setup_uv.sh --python 3.11 --track torch2`
- Pip + constraints (3.10): `pip install -e .[dev] -c constraints-3.10-compat.txt`
=======
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
>>>>>>> docs/compat-and-gallery
