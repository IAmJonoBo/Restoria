from __future__ import annotations

import os
import uuid
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .schemas import JobSpec, JobStatus, Result
from .security import apply_security


app = FastAPI(title="GFPP API", version="0.0.1")
apply_security(app)


@app.get("/healthz")
def healthz():
    info: Dict[str, Any] = {"status": "ok"}
    try:
        import torch  # type: ignore

        info["torch"] = getattr(torch, "__version__", None)
        info["cuda"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        info["torch"] = None
        info["cuda"] = False
    return JSONResponse(info)


@app.post("/restore", response_model=Result)
def restore(spec: JobSpec):
    # Synchronous single call; for batch/video, prefer /jobs + worker.
    job_id = uuid.uuid4().hex[:8]
    out_dir = os.path.join(spec.output, f"job-{job_id}")
    try:
        from src.gfpp.cli import cmd_run  # type: ignore
    except Exception:
        from gfpp.cli import cmd_run  # type: ignore

    args = [
        "--input",
        spec.input,
        "--backend",
        spec.backend,
        "--background",
        spec.background,
        "--preset",
        spec.preset,
        "--compile",
        spec.compile,
        "--metrics",
        spec.metrics,
        "--output",
        out_dir,
    ]
    if spec.seed is not None:
        args += ["--seed", str(spec.seed)]
    if spec.deterministic:
        args += ["--deterministic"]

    # Execute synchronously (in-process) for now
    code = cmd_run(args)
    status = JobStatus(id=job_id, status="done" if code == 0 else "error", result_count=0, results_path=out_dir)
    # Build a minimal result body
    return Result(job=status, metrics=[])


def main():  # pragma: no cover
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("services.api.main:app", host="0.0.0.0", port=port, reload=False)

