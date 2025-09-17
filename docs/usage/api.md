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
