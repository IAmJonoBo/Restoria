# Backend Matrix

| Engine          | Torch 2.x | ONNX Runtime | TensorRT | MPS | Notes |
|-----------------|-----------|--------------|----------|-----|-------|
| GFPGAN          | ✓         | planned      | planned  | ✓   | Baseline |
| CodeFormer      | ✓ (extra) | planned      | planned  | ✓   | Optional |
| RestoreFormer++ | ✓ (extra) | planned      | planned  | ✓   | Optional |
| Ensemble        | ✓         | –            | –        | ✓   | Weighted blend |
| Guided          | ✓         | –            | –        | ✓   | Reference‑aware |
| DiffBIR         | EXPERIMENTAL | –        | –        | –   | Heavy |
| HYPIR           | EXPERIMENTAL | –        | –        | –   | Heavy |

Background upsamplers:

| Upsampler | Torch | ONNX | Notes |
|-----------|-------|------|-------|
| RealESRGAN| ✓     | –    | Default on CUDA |
| SwinIR    | planned | –  | Optional |


## Orchestrator Routing Signals

When `--auto-backend` is enabled, routing currently considers:

- NIQE / BRISQUE (no‑ref quality) thresholds:
  - `few_artifacts`: niqe < 7.5 or brisque < 35 → stay on GFPGAN (weight preserved)
  - `heavy_degradation`: niqe ≥ 12 or brisque ≥ 55 → switch to CodeFormer
    (weight ≥ 0.6)
  - `moderate_degradation`: values in between → GFPGAN with standardized weight 0.6
- Face stats (if detector available): recorded (`face_count`, size stats)
  but not yet influencing routing (planned).
- `no_faces_detected`: routes to RestoreFormer++ when no faces are detected
  and the backend is available, improving document/scene quality.

All signals and applied rule are embedded per image under `plan.quality`,
`plan.faces`, and `plan.detail.routing_rules` in the manifest.
