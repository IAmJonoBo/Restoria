# Doctor command

The `doctor` command inspects your environment and reports available backends
and providers.

## Usage

```bash
# JSON output for automation
python -m gfpp.cli doctor --json

# Human-readable output
python -m gfpp.cli doctor
```

## What it checks

- Python version and platform
- Installed optional dependencies (e.g., torch, onnxruntime)
- Available GPU/accelerators (CUDA, MPS, DirectML if applicable)
- Registered backends and their availability
- ONNX Runtime providers and selection order

## Interpreting results

- A backend listed with `available: false` will be skipped automatically when
  selected by auto mode. You can still request a specific backend; the CLI will
  warn and fall back to GFPGAN if unavailable.
- If ONNX Runtime is present, the `ort_providers` array shows which providers
  will be attempted in order; unsupported providers are ignored.
- Warnings are prefixed with `[WARN]` and should not block restoration. They
  are informational to help you improve setup.

## Troubleshooting tips

- If GPU is not detected, ensure the correct driver and toolkit are installed
  and the Python package matches your environment.
- If ONNX Runtime is missing or only CPU is available, you can:
  - install GPU-enabled ORT for your platform, or
  - proceed with CPU or the eager backend; results will still be produced.
- When `--compile` is used and fails, the model is left unmodified and the run
  proceeds. This is expected and safe.
