# Hardware & Performance Guide

- CUDA GPUs (recommended): enables RealESRGAN background upsampling and torch.compile.
- Apple Silicon (MPS): GFPGAN works; ONNX CoreML EP planned.
- CPU-only: Works for small images; disable background upsampling for speed.

Tips:
- Use `--compile default` to JIT-compile models on Torch 2.x.
- Adjust tile size and precision (`fp16`) based on VRAM.
- Set `--seed` and `--deterministic` for reproducible runs.

