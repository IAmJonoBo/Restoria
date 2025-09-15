# UI Guide (GFPP Web)

The web app (Next.js 15 + React 19) provides a one-screen flow:

- Controls: Backend (Torch/ORT/CodeFormer/RF++), ONNX model path (for ORT), Preset, Metrics, Background, Quality, Auto Backend, Identity Lock.
- Submit: "Submit Dry-Run Job" sends a job to the local API and opens a WebSocket stream for progress.
- Queue: shows job list with progress bars, a ZIP download link when finished, and a "Re-run (dry)" button.
- Results: a gallery of before/after comparisons and per-image metrics cards. Identity lock retries display a green badge.
- Per-image Re-run: "Re-run this (dry)" submits a rerun for that specific input using the current controls.

Dev setup:

```bash
uvicorn services.api.main:app --reload --port 3000
cd apps/web && pnpm i && pnpm dev
```

Images are proxied via `/file?path=` for convenience in dev; in production, serve static files via a proper file server.

