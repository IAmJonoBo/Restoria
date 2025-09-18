# API Usage (FastAPI)

- Install extras: `pip install -e .[api]`
- Run locally with uvicorn:

  ```bash
  uvicorn services.api.main:app --reload --port 8000
  ```

- Health: `GET /healthz`
- Restore: `POST /restore` (multipart/form-data)
  - form field `files`: one or more images
  - query params: `version=1.4`, `upscale=2`, `backend=gfpgan`, `device=auto`,
    `dry_run=true`, `metrics=fast|full|off`

Notes

- Default is `dry_run=true` for smoke-safety; set `dry_run=false` for actual
  restoration.
- In non-dry-run, the server resolves weights via centralized weight logic
  (same as CLI) and runs the selected backend.
- CI runs an API smoke check as part of the workflow.
- The default deployment enforces lightweight rate limiting (~20 requests/sec,
  240/min) to protect shared hosts; tune via `services.api.security`.
- Generated outputs have EXIF metadata stripped to avoid leaking camera data in
  production downloads.
- Non-dry runs are queued and executed by a background worker so the REST API
  responds quickly even when restorers perform GPU-heavy work.

- You can opt into planner-driven routing with `auto_backend=true`; metrics include the chosen backend and reason per image.

Example response (trimmed):

```json
{
  "job": {
    "id": "job-1234",
    "status": "done"
  },
  "metrics": [
    {
      "input": "inputs/photo.png",
      "restored_img": "results/job-1234/photo.png",
      "plan": {
        "backend": "codeformer",
        "reason": "auto_route",
        "confidence": 0.82,
        "params": {
          "weight": 0.7
        }
      },
      "metrics": {
        "runtime_sec": 2.31,
        "plan_backend": "codeformer"
      }
    }
  ]
}
```
