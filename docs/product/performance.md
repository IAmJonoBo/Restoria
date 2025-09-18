# Performance Guide

This guide summarises the knobs that have the biggest impact on runtime and how to measure them.

## Quick wins

- **Use the planner** – `restoria run --auto` or `auto_backend=true` on the API will pick the fastest backend
  that still meets quality targets and skips heavyweight models when unnecessary.
- **Scale with hardware** – on CUDA devices pass `--device cuda`; on Apple Silicon use the default `auto`
  which resolves to `mps`; fall back to `cpu` only for validation.
- **Prefer cached weights** – call `restoria doctor --download` or use the `scripts/cache_weights.py` helper
  from the release playbook to avoid repeated downloads when benchmarking.

## Torch and ONNX Runtime

| Scenario                    | Recommended flag                           | Notes |
| --------------------------- | ----------------------------------------- | ----- |
| Torch eager (default)       | _none_                                     | Lowest startup overhead; best for small batches. |
| Torch compile (Ampere+)     | `--compile`                                | Uses `torch.compile`, adds a warm-up compile step but often halves runtime after the first batch. |
| ONNX Runtime CUDA           | `--backend gfpgan-ort --ort-providers CUDAExecutionProvider TensorRTExecutionProvider` | Requires `onnxruntime-gpu`; best latency for batch workloads. |
| ONNX Runtime CPU            | `--backend gfpgan-ort --ort-providers CPUExecutionProvider` | Faster than eager on CPU-only hosts; avoid on GPU machines. |

On the API, the same flags are supported via `JobSpec.compile` (set to anything other than `none`) and
`JobSpec.auto_backend`/`JobSpec.metrics`/`JobSpec.background`.

## Benchmarking

1. Collect a representative folder of inputs (mixed lighting, resolution, and face counts).
2. Run the light benchmarking harness to capture per-backend telemetry:
   ```bash
   restoria bench --input inputs/portraits --output bench/out --backend gfpgan --metrics fast
   ```
3. Re-run with the alternative backend or flag you want to compare (`--backend gfpgan-ort`, `--compile`, etc.).
4. Each run emits `metrics.json` with `runtime_sec`, planner decisions, and quality metrics.  Compare the JSON files or
   load them into the provided `notebooks/Restoria_Benchmark.ipynb` quick-start.

For automation, the CI job `bench-smoke.yml` runs a smoke-sized run on every PR and uploads the raw metrics artifact.

## Provider matrix

| Hardware         | Recommended providers                                      | Fallbacks |
| ---------------- | ---------------------------------------------------------- | --------- |
| NVIDIA (CUDA 11) | `CUDAExecutionProvider`, then `TensorRTExecutionProvider`  | `CPUExecutionProvider` |
| NVIDIA (legacy)  | `CUDAExecutionProvider`                                    | `CPUExecutionProvider` |
| CPU only         | `CPUExecutionProvider`                                     | – |
| Apple Silicon    | (Torch eager with `--device auto`; ORT MPS is experimental) | – |

## Capturing telemetry

Set `RESTORIA_PROFILE=1` before invoking the CLI to append per-image timing to `metrics.json`.  The job manager also
records the planner summary in every API response, so telemetry is available whether you run via CLI or the FastAPI
service.

For long-running workloads, consider running under `python -m cProfile -o profile.out restoria run ...` or using the
`bench` command to collect a canonical CSV/HTML report.

## Troubleshooting

- **Compile path slower than eager** – the first batch includes the compilation cost.  Warm up with a single dry run or
  enable the planner so compile is used only on heavy inputs.
- **ORT falls back to CPU** – check `restoria doctor --json` or `/doctor?format=json` in the API; ensure the CUDA/TensorRT
  providers are installed and listed first.
- **MPS accuracy issues** – for critical work, prefer running on CUDA until Apple releases the next MPS update.

