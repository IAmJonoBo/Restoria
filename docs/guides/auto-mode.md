# Auto mode (experimental, opt-in)

Auto mode selects a backend and parameters per image using lightweight probes.
It is disabled by default and can be enabled with:

- CLI: `gfpup run --auto --input <path> --output out/`
- API/UI: toggle the auto backend option

What it does (transparent rules):

- Probes image quality (NIQE, BRISQUE where available)
- Routes to GFPGAN (clean), CodeFormer (more fidelity on strong degradation),
  or keeps your choice
- Suggests background upsampling (Real-ESRGAN by default) when global quality
  remains low

Advanced metrics (optional)

- If installed, advanced no-reference IQA (e.g., MANIQA/CONTRIQUE) may be
  recorded by the planner.
- These are purely informational today and do not change routing unless we add
  explicit opt-in rules.

Ensemble mode

- Auto mode never selects ensemble implicitly. Use `--backend ensemble`
  explicitly to enable blending.

Notes

- Probes are best-effort and skipped if dependencies are missing. Behavior
  gracefully falls back to your chosen backend.
- Identity guardrails are part of the metrics/identity lock path and remain opt-in.

TODO

- Incorporate face size/count signals and richer thresholds
- Show the plan explanation in the UI
