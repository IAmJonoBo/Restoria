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
