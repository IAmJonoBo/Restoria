# GFPP Quickstart

Install (editable) with optional extras:

```bash
pip install -e ".[dev,metrics,arcface,codeformer,restoreformerpp,ort,web]"
```

Run the new CLI (GFPGAN baseline):

```bash
gfpup run --input inputs/whole_imgs --backend gfpgan --metrics fast --output out/
# Optional flags:
#   --dry-run (copy inputs → outputs, no model load)
#   --quality {quick|balanced|best} (maps BG SR tile/precision)
#   --auto-backend (choose engine per-image)
#   --identity-lock --identity-threshold 0.25
#   --optimize --weights-cand "0.3,0.5,0.7"
#   --compile {none,default,max} (optional torch.compile path)

Plan & Metrics:

- When `--auto-backend` is enabled each output record embeds a `plan` block:

```json
{
  "plan": {
    "backend": "gfpgan",
    "reason": "moderate_degradation",
    "quality": { "niqe": 11.2, "brisque": 48.5 },
    "faces": { "face_count": 1 },
    "detail": {
      "routing_rules": {
        "few_artifacts": "niqe < 7.5 or brisque < 35",
        "heavy_degradation": "niqe >= 12 or brisque >= 55",
        "moderate_degradation": "otherwise"
      },
      "decision_inputs": { "niqe": 11.2, "brisque": 48.5, "face_count": 1 }
    }
  }
}
```

- Dry-run still writes a manifest and now computes no‑reference quality metrics (NIQE / BRISQUE) when `--metrics` is not `off`.
- If a probe or metric dependency is missing the corresponding values are omitted or set to `null` without failing the run.
```

API dev server:

```bash
uvicorn services.api.main:app --reload
# REST endpoints:
#   POST /jobs (JobSpec)
#   GET  /jobs, /jobs/:id, /results/:id
#   WS   /jobs/:id/stream (status/image/manifest/eof events)
#   POST /jobs/:id/rerun (overrides)
```

Web (dev):

```bash
cd apps/web
pnpm i
pnpm dev
```
