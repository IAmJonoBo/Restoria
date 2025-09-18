from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from src.restoria.core import planner as rest_planner  # type: ignore
from src.restoria.io.manifest import RunManifest, write_manifest  # type: ignore
from src.restoria.io.schemas import MetricsPayload, ResultRecord, plan_summary_from

if TYPE_CHECKING:  # pragma: no cover
    from .schemas import JobSpec  # type: ignore
else:
    JobSpec = Any  # type: ignore

INPUT_IMAGE_NAME = "in.png"
OUTPUT_IMAGE_NAME = "out.png"


def _scrub_exif(path: str) -> None:
    """Remove EXIF metadata in-place when piexif is available."""

    try:
        import piexif  # type: ignore

        piexif.remove(path)  # type: ignore[arg-type]
    except ImportError:
        pass
    except Exception:
        # Do not fail the job if scrubbing fails; log when logging is available.
        pass


@dataclass
class _Job:
    id: str
    spec: JobSpec
    status: str = "queued"
    progress: float = 0.0
    result_count: int = 0
    results_path: str | None = None
    error: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    events: asyncio.Queue = field(default_factory=asyncio.Queue)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    def create(self, spec: JobSpec) -> _Job:
        jid = uuid.uuid4().hex[:8]
        job = _Job(id=jid, spec=spec)
        out_dir = os.path.join(spec.output, f"job-{jid}")
        os.makedirs(out_dir, exist_ok=True)
        job.results_path = out_dir
        self._jobs[jid] = job
        return job

    async def enqueue(self, jid: str) -> None:
        await self._queue.put(jid)

    def start(self) -> None:
        if self._worker_task is None:
            loop = asyncio.get_running_loop()
            self._worker_task = loop.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:  # pragma: no cover - shutdown path
                pass
            self._worker_task = None

    async def _worker(self) -> None:
        while True:
            jid = await self._queue.get()
            try:
                await self.run(jid)
            except Exception as exc:  # pragma: no cover - defensive
                job = self._jobs.get(jid)
                if job:
                    job.status = "error"
                    job.error = str(exc)
                    await job.events.put({"type": "status", "status": job.status, "error": job.error})
            finally:
                self._queue.task_done()

    def get(self, jid: str) -> _Job | None:
        return self._jobs.get(jid)

    def list(self) -> list[_Job]:
        return list(self._jobs.values())

    async def run(self, jid: str) -> None:
        job = self._jobs[jid]
        job.status = "running"
        await job.events.put({"type": "status", "status": job.status})
        try:
            from src.gfpp.io.loading import list_inputs, load_image_bgr, save_image  # type: ignore

            spec = job.spec
            inputs = list_inputs(spec.input)
            total_inputs = max(1, len(inputs))
            self._set_deterministic(spec.seed, spec.deterministic)

            planner_opts: dict[str, Any] = {
                "backend": spec.backend,
                "auto": bool(getattr(spec, "auto_backend", False)),
            }
            compile_hint = str(getattr(spec, "compile", "")).strip().lower()
            if compile_hint and compile_hint not in {"none", "off"}:
                planner_opts["compile"] = True

            records: list[ResultRecord] = []
            is_smoke = spec.dry_run or os.environ.get("NB_CI_SMOKE") == "1"

            if is_smoke:
                max_items = int(os.environ.get("GFPP_API_DRYRUN_MAX", "4"))
                for idx, pth in enumerate(inputs[:max_items], start=1):
                    t0 = time.time()
                    plan = await asyncio.to_thread(rest_planner.compute_plan, pth, planner_opts)
                    plan_summary = plan_summary_from(plan)
                    img = await asyncio.to_thread(load_image_bgr, pth)
                    if img is None:
                        await job.events.put({"type": "warn", "msg": f"Failed to read: {pth}"})
                        continue
                    base = os.path.splitext(os.path.basename(pth))[0]
                    out_img = os.path.join(job.results_path or spec.output, f"{base}.png")
                    await asyncio.to_thread(save_image, out_img, img)
                    _scrub_exif(out_img)
                    metrics = {
                        "runtime_sec": time.time() - t0,
                        "plan_backend": plan.backend,
                        "plan_reason": plan.reason,
                        "plan_confidence": plan.confidence,
                    }
                    record = ResultRecord(
                        input=pth,
                        backend=plan.backend,
                        restored_img=out_img,
                        metrics=metrics,
                        plan=plan_summary,
                    )
                    records.append(record)
                    job.result_count = idx
                    job.progress = idx / total_inputs
                    await job.events.put(
                        {"type": "image", **record.model_dump(exclude_none=True), "progress": job.progress}
                    )
            else:
                from src.gfpp.background import build_realesrgan  # type: ignore
                from src.gfpp.metrics import ArcFaceIdentity, LPIPSMetric  # type: ignore
                from src.restoria.core.registry import get as load_backend  # type: ignore

                if spec.background == "realesrgan":
                    if spec.quality == "quick":
                        tile, precision = 0, "fp16"
                    elif spec.quality == "best":
                        tile, precision = 0, "fp32"
                    else:
                        tile, precision = 400, "auto"
                    bg = await asyncio.to_thread(build_realesrgan, device="cuda", tile=tile, precision=precision)
                else:
                    bg = None

                rest_cache: dict[str, Any] = {}

                def get_restorer(name: str):
                    if name not in rest_cache:
                        try:
                            rest_cls = load_backend(name)
                        except KeyError:
                            rest_cls = load_backend("gfpgan")
                            name = "gfpgan"
                        kwargs: dict[str, Any] = {"device": "auto"}
                        if bg is not None:
                            kwargs["bg_upsampler"] = bg
                        rest_cache[name] = rest_cls(**kwargs)
                    return rest_cache[name]

                preset_weight = {"natural": 0.5, "detail": 0.7, "document": 0.3}.get(spec.preset, 0.5)
                cfg_base: dict[str, Any] = {
                    "version": "1.4",
                    "upscale": 2,
                    "use_parse": True,
                    "detector": "retinaface_resnet50",
                    "weight": preset_weight,
                    "no_download": False,
                }
                if spec.backend == "gfpgan-ort" and spec.model_path_onnx:
                    cfg_base["model_path_onnx"] = spec.model_path_onnx

                arc = ArcFaceIdentity(no_download=True) if spec.metrics in {"fast", "full"} else None
                lpips = LPIPSMetric() if spec.metrics == "full" else None

                for idx, pth in enumerate(inputs, start=1):
                    t0 = time.time()
                    plan = await asyncio.to_thread(rest_planner.compute_plan, pth, planner_opts)
                    plan_summary = plan_summary_from(plan)
                    img = await asyncio.to_thread(load_image_bgr, pth)
                    if img is None:
                        await job.events.put({"type": "warn", "msg": f"Failed to read: {pth}"})
                        continue

                    restorer = get_restorer(plan.backend)
                    cfg_img = dict(cfg_base)
                    cfg_img["input_path"] = pth
                    if "weight" in plan.params:
                        try:
                            cfg_img["weight"] = float(plan.params["weight"])
                        except Exception:
                            cfg_img["weight"] = plan.params["weight"]
                    if "detector" in plan.params:
                        cfg_img["detector"] = plan.params["detector"]
                    if plan.backend == "gfpgan-ort" and spec.model_path_onnx:
                        cfg_img["model_path_onnx"] = spec.model_path_onnx

                    result = None
                    if spec.optimize:
                        try:
                            candidates = [
                                min(max(float(x.strip()), 0.0), 1.0) for x in spec.weights_cand.split(",") if x.strip()
                            ]
                        except Exception:
                            candidates = [0.3, 0.5, 0.7]
                        import cv2
                        import tempfile

                        best_score = None
                        best_record = None
                        for weight in candidates:
                            cfg_try = dict(cfg_img)
                            cfg_try["weight"] = weight
                            candidate = await asyncio.to_thread(restorer.restore, img, cfg_try)
                            score = None
                            if arc and arc.available():
                                tmp = tempfile.mkdtemp()
                                a = os.path.join(tmp, INPUT_IMAGE_NAME)
                                b = os.path.join(tmp, OUTPUT_IMAGE_NAME)
                                cv2.imwrite(a, img)
                                cv2.imwrite(
                                    b,
                                    candidate.restored_image
                                    if (candidate and candidate.restored_image is not None)
                                    else img,
                                )
                                score = arc.cosine_from_paths(a, b)
                            if score is None and lpips and lpips.available():
                                tmp = tempfile.mkdtemp()
                                a = os.path.join(tmp, INPUT_IMAGE_NAME)
                                b = os.path.join(tmp, OUTPUT_IMAGE_NAME)
                                cv2.imwrite(a, img)
                                cv2.imwrite(
                                    b,
                                    candidate.restored_image
                                    if (candidate and candidate.restored_image is not None)
                                    else img,
                                )
                                distance = lpips.distance_from_paths(a, b)
                                score = -distance if isinstance(distance, float) else None
                            if score is None:
                                try:
                                    score = -abs(weight - float(preset_weight))
                                except Exception:
                                    score = -1.0
                            if best_score is None or (score is not None and score > best_score):
                                best_score = score
                                best_record = candidate
                                cfg_img["weight"] = weight
                        result = best_record if best_record is not None else restorer.restore(img, cfg_img)
                    else:
                        result = await asyncio.to_thread(restorer.restore, img, cfg_img)

                    base = os.path.splitext(os.path.basename(pth))[0]
                    out_img = os.path.join(job.results_path or spec.output, f"{base}.png")
                    if result and result.restored_image is not None:
                        await asyncio.to_thread(save_image, out_img, result.restored_image)
                    else:
                        await asyncio.to_thread(save_image, out_img, img)
                    _scrub_exif(out_img)
                    metrics = {
                        "runtime_sec": time.time() - t0,
                        "plan_backend": plan.backend,
                        "plan_reason": plan.reason,
                        "plan_confidence": plan.confidence,
                    }
                    if arc and arc.available():
                        metrics["arcface_cosine"] = arc.cosine_from_paths(pth, out_img)
                    if lpips and lpips.available():
                        metrics["lpips_alex"] = lpips.distance_from_paths(pth, out_img)

                    if spec.identity_lock and arc and arc.available():
                        import cv2
                        import tempfile

                        tmp = tempfile.mkdtemp()
                        a = os.path.join(tmp, INPUT_IMAGE_NAME)
                        b = os.path.join(tmp, OUTPUT_IMAGE_NAME)
                        cv2.imwrite(a, img)
                        baseline_img = result.restored_image if (result and result.restored_image is not None) else img
                        cv2.imwrite(b, baseline_img)
                        baseline = arc.cosine_from_paths(a, b)
                        if baseline is not None and baseline < float(spec.identity_threshold):
                            cfg_retry = dict(cfg_img)
                            cfg_retry["weight"] = max(0.2, float(cfg_img.get("weight", preset_weight)) - 0.2)
                            retry = await asyncio.to_thread(restorer.restore, img, cfg_retry)
                            retry_img = retry.restored_image if (retry and retry.restored_image is not None) else img
                            cv2.imwrite(b, retry_img)
                            improved = arc.cosine_from_paths(a, b)
                            if improved is not None and improved > (baseline or 0):
                                result = retry
                                await asyncio.to_thread(
                                    save_image,
                                    out_img,
                                    retry.restored_image if retry and retry.restored_image is not None else img,
                                )
                                metrics["identity_retry"] = True

                    record = ResultRecord(
                        input=pth,
                        backend=plan.backend,
                        restored_img=out_img,
                        metrics=metrics,
                        plan=plan_summary,
                    )
                    records.append(record)
                    job.result_count = idx
                    job.progress = idx / total_inputs
                    await job.events.put(
                        {"type": "image", **record.model_dump(exclude_none=True), "progress": job.progress}
                    )

            job.results = [record.model_dump(exclude_none=True) for record in records]

            out_dir = job.results_path or spec.output
            if hasattr(spec, "model_dump"):
                args_payload = spec.model_dump()
            elif hasattr(spec, "__dict__"):
                args_payload = dict(spec.__dict__)
            else:
                args_payload = asdict(spec)

            manifest = RunManifest(args=args_payload, device="auto", results=records)
            write_manifest(os.path.join(out_dir, "manifest.json"), manifest)
            await job.events.put({"type": "manifest", "path": os.path.join(out_dir, "manifest.json")})

            metrics_payload = MetricsPayload(metrics=records, plan=records[0].plan if records else None)

            def write_metrics() -> None:
                import json as _json

                with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
                    _json.dump(metrics_payload.model_dump(exclude_none=True), fh, indent=2)

            await asyncio.to_thread(write_metrics)

            try:
                from src.gfpp.reports.html import write_html_report  # type: ignore

                write_html_report(out_dir, job.results, path=os.path.join(out_dir, "report.html"))
            except Exception:
                pass

            job.status = "done"
            await job.events.put({"type": "status", "status": job.status, "results_path": job.results_path})
        except Exception as exc:  # pragma: no cover
            job.status = "error"
            job.error = str(exc)
            await job.events.put({"type": "status", "status": job.status, "error": job.error})
        finally:
            await job.events.put({"type": "eof"})

    async def stream(self, jid: str):
        job = self._jobs[jid]
        while True:
            msg = await job.events.get()
            yield msg
            if msg.get("type") == "eof":
                break

    @staticmethod
    def _set_deterministic(seed: int | None, deterministic: bool) -> None:
        try:
            import random

            if seed is not None:
                random.seed(seed)
        except Exception:
            pass
        try:
            import numpy as np  # type: ignore

            if seed is not None:
                numpy_seed = int(seed)
                np.random.seed(numpy_seed)
        except Exception:
            pass
        try:
            import torch  # type: ignore

            if seed is not None:
                torch.manual_seed(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass


manager = JobManager()
