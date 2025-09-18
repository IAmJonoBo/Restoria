from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from .jobs import manager
from .schemas import JobSpec, JobStatus, Result, MetricCard
from .security import apply_security, rate_limit

# Error message constants
NOT_FOUND_MSG = "not found"

app = FastAPI(title="GFPP API", version="0.0.1")
apply_security(app)


@app.on_event("startup")
async def _startup() -> None:
    manager.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await manager.stop()


@app.get("/healthz")
def healthz():
    info: dict[str, Any] = {"status": "ok"}
    try:
        import torch  # type: ignore

        info["torch"] = getattr(torch, "__version__", None)
        info["cuda"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        info["torch"] = None
        info["cuda"] = False
    return JSONResponse(info)


@app.post("/restore", response_model=Result)
@rate_limit("30/minute")
async def restore(spec: JobSpec):
    job = manager.create(spec)
    await manager.run(job.id)
    status = JobStatus(
        id=job.id,
        status=job.status,
        progress=job.progress,
        result_count=job.result_count,
        results_path=job.results_path,
        error=job.error,
    )
    cards = [
        MetricCard(
            input=entry.get("input"),
            restored_img=entry.get("restored_img"),
            metrics=entry.get("metrics", {}),
            plan=entry.get("plan", {}),
        )
        for entry in job.results
    ]
    return Result(job=status, metrics=cards)


@app.post("/jobs")
@rate_limit("60/minute")
async def submit_job(spec: JobSpec):
    job = manager.create(spec)
    if spec.dry_run:
        # Synchronous, fast path for dry-run jobs: complete before returning
        await manager.run(job.id)
    else:
        await manager.enqueue(job.id)

    return JobStatus(
        id=job.id,
        status=job.status,
        progress=job.progress,
        result_count=job.result_count,
        results_path=job.results_path,
    )


@app.get("/jobs")
@rate_limit("120/minute")
async def list_jobs():
    items = []
    for j in manager.list():
        items.append(
            JobStatus(
                id=j.id,
                status=j.status,
                progress=j.progress,
                result_count=j.result_count,
                results_path=j.results_path,
                error=j.error,
            )
        )
    return items


@app.get("/jobs/{job_id}")
@rate_limit("120/minute")
async def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        return JSONResponse({"error": NOT_FOUND_MSG}, status_code=404)
    return JobStatus(
        id=job.id,
        status=job.status,
        progress=job.progress,
        result_count=job.result_count,
        results_path=job.results_path,
        error=job.error,
    )


@app.post("/jobs/{job_id}/rerun")
@rate_limit("60/minute")
async def rerun_job(job_id: str, overrides: dict | None = None):
    """Re-run a job with optional overrides to its original spec.

    Example overrides: {"preset": "detail", "metrics": "full", "dry_run": true}
    """
    job = manager.get(job_id)
    if not job:
        return JSONResponse({"error": NOT_FOUND_MSG}, status_code=404)
    base = job.spec.model_dump()  # pydantic v2
    overrides = overrides or {}
    # Only allow known keys
    allowed = {
        "input",
        "backend",
        "background",
        "quality",
        "preset",
        "compile",
        "seed",
        "deterministic",
        "metrics",
        "output",
        "dry_run",
        "model_path_onnx",
        "auto_backend",
    }
    new_spec = {**base, **{k: v for k, v in overrides.items() if k in allowed}}
    # Create new job
    from .schemas import JobSpec as _JobSpec

    j = manager.create(_JobSpec(**new_spec))
    await manager.enqueue(j.id)

    return JobStatus(
        id=j.id, status=j.status, progress=j.progress, result_count=j.result_count, results_path=j.results_path
    )


@app.websocket("/jobs/{job_id}/stream")
async def ws_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = manager.get(job_id)
    if not job:
        await websocket.send_json({"error": NOT_FOUND_MSG})
        await websocket.close(code=1000)


@app.get("/results/{job_id}")
async def download_results(job_id: str):
    import shutil

    job = manager.get(job_id)
    if not job or not job.results_path:
        return JSONResponse({"error": NOT_FOUND_MSG}, status_code=404)
    base = os.path.abspath(job.results_path)
    parent = os.path.dirname(base)
    zip_base = os.path.join(parent, f"{os.path.basename(base)}")
    zip_path = f"{zip_base}.zip"
    if not os.path.exists(zip_path):
        shutil.make_archive(zip_base, "zip", base)
    return FileResponse(zip_path, filename=os.path.basename(zip_path))


@app.get("/file")
async def get_file(path: str):
    # Minimal dev-time file serving for inputs/results. In production, serve via a proper static server.
    abspath = os.path.abspath(path)
    if not os.path.isfile(abspath):
        return JSONResponse({"error": NOT_FOUND_MSG}, status_code=404)
    # Basic path guard: only allow files under project directory
    proj = os.path.abspath(os.getcwd())
    if not abspath.startswith(proj):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return FileResponse(abspath)


__all__ = [
    "app",
    "restore",
    "submit_job",
    "list_jobs",
    "get_job",
    "rerun_job",
    "download_results",
    "ws_stream",
    "get_file",
]
