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
