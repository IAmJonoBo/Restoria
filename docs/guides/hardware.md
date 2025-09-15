# Performance on your hardware

This project keeps the default backend and outputs unchanged, while providing optional fast paths.

Fast paths (opt-in):

- torch.compile: `--compile default|max` to JIT optimize models. Falls back safely on error.
- ONNX Runtime: CPU is default; if installed, we auto-select the best available EP (CUDA → TensorRT → DirectML → CoreML → CPU).

Tips:

- Use `gfpup doctor` to see Torch, CUDA and ORT providers on your machine.
- For laptops with both iGPU/dGPU, ensure the discrete GPU is selected.
- For CPU-only runs, set `--device cpu` and prefer `--quality quick`.

TODO

- Add per-backend export helpers and quantization notes.
