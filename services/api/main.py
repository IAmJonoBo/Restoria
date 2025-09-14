from __future__ import annotations

import os
import uuid
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse

from .schemas import JobSpec, JobStatus, Result
from .security import apply_security
from .jobs import manager


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


@app.post("/jobs")
async def submit_job(spec: JobSpec):
    job = manager.create(spec)
    # Fire-and-forget: schedule in background
    import asyncio

    asyncio.create_task(manager.run(job.id))
    return JobStatus(id=job.id, status=job.status, progress=job.progress, result_count=job.result_count, results_path=job.results_path)


@app.get("/jobs")
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
async def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JobStatus(id=job.id, status=job.status, progress=job.progress, result_count=job.result_count, results_path=job.results_path, error=job.error)


@app.websocket("/jobs/{job_id}/stream")
async def ws_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = manager.get(job_id)
    if not job:
        await websocket.send_json({"error": "not found"})
        await websocket.close(code=1000)
        return


@app.get("/results/{job_id}")
async def download_results(job_id: str):
    import shutil

    job = manager.get(job_id)
    if not job or not job.results_path:
        return JSONResponse({"error": "not found"}, status_code=404)
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
        return JSONResponse({"error": "not found"}, status_code=404)
    # Basic path guard: only allow files under project directory
    proj = os.path.abspath(os.getcwd())
    if not abspath.startswith(proj):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return FileResponse(abspath)
    try:
        async for evt in manager.stream(job_id):
            await websocket.send_json(evt)
    except WebSocketDisconnect:  # pragma: no cover
        return


def main():  # pragma: no cover
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("services.api.main:app", host="0.0.0.0", port=port, reload=False)
