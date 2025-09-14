# GFPP Web (Next.js 15 + React 19)

This is a scaffold for the premium web UI. It targets:
- Upload/Queue/Results single-page flow
- Accessible before/after compare slider
- Streaming progress via SSE/WS
- Presets + per-image overrides

Commands (dev):

```
cd apps/web
pnpm i
pnpm dev
```

The app expects the API to be available at the same origin (proxy in dev). For a quick smoke, run:

```
uvicorn services.api.main:app --reload --port 3000
# then in another shell
cd apps/web && pnpm dev
```

This will let the Next.js app call the API/WS via relative URLs.
