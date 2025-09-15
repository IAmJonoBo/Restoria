# Auto mode (experimental, opt-in)

Auto mode selects a backend and parameters per image using lightweight probes. It is disabled by default and can be enabled with:

- CLI: `gfpup run --auto --input <path> --output out/`
- API/UI: toggle the auto backend option

What it does (transparent rules):

- Probes image quality (NIQE, BRISQUE where available)
- Routes to GFPGAN (clean), CodeFormer (more fidelity on strong degradation), or keeps your choice
- Suggests background upsampling (Real-ESRGAN by default) when global quality remains low

## Decision guide (current rules)

These conservative rules are used when quality probes are available:

| Quality signal (approx.) | NIQE | BRISQUE | Backend | Weight tweak | Background |
|---|---:|---:|---|---:|---|
| Clean/Good | ≤ 7 | ≤ 35 | GFPGAN | keep user value | keep user value |
| Moderate | 7–10 | 35–50 | GFPGAN | max(0.6, user) | keep user value |
| Heavy degradation | > 10 | > 50 | CodeFormer | 0.7–0.9 | suggest realesrgan |

Notes:

- When NIQE and BRISQUE disagree, NIQE takes precedence for now.
- If probes are unavailable, your selected backend/params are kept unchanged.

Notes

- Probes are best-effort and skipped if dependencies are missing. Behavior gracefully falls back to your chosen backend.
- Identity guardrails are part of the metrics/identity lock path and remain opt-in.

TODO

- Incorporate face size/count signals and richer thresholds
- Show the plan explanation in the UI
- Add identity guardrails option and decision table alignment
