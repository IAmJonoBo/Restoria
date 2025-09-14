# Backend Matrix

| Engine          | Torch 2.x | ONNX Runtime | TensorRT | MPS | Notes |
|-----------------|-----------|--------------|----------|-----|-------|
| GFPGAN          | ✓         | planned      | planned  | ✓   | Baseline |
| CodeFormer      | ✓ (extra) | planned      | planned  | ✓   | Optional |
| RestoreFormer++ | ✓ (extra) | planned      | planned  | ✓   | Optional |
| DiffBIR         | EXPERIMENTAL | –        | –        | –   | Heavy |
| HYPIR           | EXPERIMENTAL | –        | –        | –   | Heavy |

Background upsamplers:

| Upsampler | Torch | ONNX | Notes |
|-----------|-------|------|-------|
| RealESRGAN| ✓     | –    | Default on CUDA |
| SwinIR    | planned | –  | Optional |

