# ONNX Runtime Guide (GFPP)

This guide explains how to run GFPGAN with ONNX Runtime.

- Install ORT:
  - CPU: `pip install onnxruntime`
  - CUDA/TensorRT: `pip install onnxruntime-gpu` (and ensure CUDA/TensorRT installed)

- Export ONNX graph (outline):
  - The CLI has a stub:
    `gfpup export-onnx --version 1.4 --model-path /path/GFPGANv1.4.pth --out gfpgan.onnx`
    (alias: `--output gfpgan.onnx`)
  - Implement export by initializing `restorer.gfpgan` and calling `torch.onnx.export`
    with a 512x512 BCHW normalized dummy input. Validate with ORT.

- Run ORT backend:
  - CLI: `gfpup run --input inputs/whole_imgs --backend gfpgan-ort --model-path-onnx
    /path/gfpgan.onnx --output out/`
  - API: pass `{\"backend\":\"gfpgan-ort\",\"model_path_onnx\":\"/path/gfpgan.onnx\"}`
    in the JobSpec.

- Providers:
  - The runtime logs available providers and the selected provider; metrics include
    `ort_provider` and `ort_init_sec`.

- Fallback:
  - If ORT initialization or inference is not available, the system falls back to
    the Torch backend transparently and records `backend: onnxruntime+torch-fallback`.

