from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .schemas import JobSpec, JobStatus


@dataclass
class _Job:
    id: str
    spec: JobSpec
    status: str = "queued"  # queued|running|done|error
    progress: float = 0.0
    result_count: int = 0
    results_path: Optional[str] = None
    error: Optional[str] = None
    events: asyncio.Queue = field(default_factory=asyncio.Queue)


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, _Job] = {}

    def create(self, spec: JobSpec) -> _Job:
        jid = uuid.uuid4().hex[:8]
        j = _Job(id=jid, spec=spec)
        # results dir per job
        out_dir = os.path.join(spec.output, f"job-{jid}")
        os.makedirs(out_dir, exist_ok=True)
        j.results_path = out_dir
        self._jobs[jid] = j
        return j

    def get(self, jid: str) -> Optional[_Job]:
        return self._jobs.get(jid)

    def list(self) -> List[_Job]:
        # Return jobs in creation order
        return list(self._jobs.values())

    async def run(self, jid: str) -> None:
        job = self._jobs[jid]
        job.status = "running"
        await job.events.put({"type": "status", "status": job.status})
        try:
            # Import locally to avoid global heavy imports
            from src.gfpp.io.loading import list_inputs, load_image_bgr, save_image  # type: ignore
            from src.gfpp.io.manifest import RunManifest, write_manifest  # type: ignore

            spec = job.spec
            inputs = list_inputs(spec.input)
            n = max(1, len(inputs))
            # Determinism
            self._set_deterministic(spec.seed, spec.deterministic)

            results_rec: List[Dict[str, Any]] = []
            # Dry-run or smoke mode: skip heavy model loading
            if spec.dry_run or os.environ.get("NB_CI_SMOKE") == "1":
                count = 0
                for pth in inputs:
                    t0 = time.time()
                    img = load_image_bgr(pth)
                    if img is None:
                        await job.events.put({"type": "warn", "msg": f"Failed to read: {pth}"})
                        continue
                    base = os.path.splitext(os.path.basename(pth))[0]
                    out_img = os.path.join(job.results_path or spec.output, f"{base}.png")
                    save_image(out_img, img)
                    metrics = {"runtime_sec": time.time() - t0}
                    count += 1
                    job.result_count = count
                    job.progress = count / n
                    results_rec.append({"input": pth, "restored_img": out_img, "metrics": metrics})
                    await job.events.put({"type": "image", "input": pth, "output": out_img, "metrics": metrics, "progress": job.progress})
            else:
                # Heavy path: perform real restoration
                from src.gfpp.restorers.gfpgan import GFPGANRestorer  # type: ignore
                from src.gfpp.restorers.gfpgan_ort import ORTGFPGANRestorer  # type: ignore
                from src.gfpp.restorers.codeformer import CodeFormerRestorer  # type: ignore
                from src.gfpp.restorers.restoreformerpp import RestoreFormerPP  # type: ignore
                from src.gfpp.background import build_realesrgan  # type: ignore
                from src.gfpp.metrics import ArcFaceIdentity, LPIPSMetric  # type: ignore

                if spec.background == "realesrgan":
                    if spec.quality == "quick":
                        tile, prec = 0, "fp16"
                    elif spec.quality == "best":
                        tile, prec = 0, "fp32"
                    else:
                        tile, prec = 400, "auto"
                    bg = build_realesrgan(device="cuda", tile=tile, precision=prec)
                else:
                    bg = None
                # Restorer selection
                if spec.backend == "gfpgan":
                    rest = GFPGANRestorer(device="auto", bg_upsampler=bg)
                elif spec.backend == "gfpgan-ort":
                    rest = ORTGFPGANRestorer(device="auto", bg_upsampler=bg)
                elif spec.backend == "codeformer":
                    rest = CodeFormerRestorer(device="auto", bg_upsampler=bg)
                elif spec.backend == "restoreformerpp":
                    rest = RestoreFormerPP(device="auto", bg_upsampler=bg)
                else:
                    rest = GFPGANRestorer(device="auto", bg_upsampler=bg)
                # Presets -> weight mapping
                preset_weight = {"natural": 0.5, "detail": 0.7, "document": 0.3}.get(spec.preset, 0.5)

                cfg: Dict[str, Any] = {
                    "version": "1.4",
                    "upscale": 2,
                    "use_parse": True,
                    "detector": "retinaface_resnet50",
                    "weight": preset_weight,
                    "no_download": False,
                }
                arc = ArcFaceIdentity(no_download=True) if spec.metrics in {"fast", "full"} else None
                lpips = LPIPSMetric() if spec.metrics == "full" else None

                count = 0
                for pth in inputs:
                    t0 = time.time()
                    img = load_image_bgr(pth)
                    if img is None:
                        await job.events.put({"type": "warn", "msg": f"Failed to read: {pth}"})
                        continue
                    cfg["input_path"] = pth
                    # Pass through model_path_onnx when present
                    if isinstance(rest, ORTGFPGANRestorer) and spec.model_path_onnx:
                        cfg["model_path_onnx"] = spec.model_path_onnx
                    res = rest.restore(img, cfg)
                    base = os.path.splitext(os.path.basename(pth))[0]
                    out_img = os.path.join(job.results_path or spec.output, f"{base}.png")
                    if res and res.restored_image is not None:
                        save_image(out_img, res.restored_image)
                    else:
                        save_image(out_img, img)
                    metrics = {"runtime_sec": time.time() - t0}
                    if arc and arc.available():
                        metrics["arcface_cosine"] = arc.cosine_from_paths(pth, out_img)
                    if lpips and lpips.available():
                        metrics["lpips_alex"] = lpips.distance_from_paths(pth, out_img)
                    count += 1
                    job.result_count = count
                    job.progress = count / n
                    results_rec.append({"input": pth, "restored_img": out_img, "metrics": metrics})
                    await job.events.put({"type": "image", "input": pth, "output": out_img, "metrics": metrics, "progress": job.progress})

            # Write manifest.json for reproducibility
            try:
                man = RunManifest(args=asdict(spec), device="auto", results=results_rec)
                write_manifest(os.path.join(job.results_path or spec.output, "manifest.json"), man)
                await job.events.put({"type": "manifest", "path": os.path.join(job.results_path or spec.output, "manifest.json")})
            except Exception:
                pass

            job.status = "done"
            await job.events.put({"type": "status", "status": job.status, "results_path": job.results_path})
        except Exception as e:  # pragma: no cover
            job.status = "error"
            job.error = str(e)
            await job.events.put({"type": "status", "status": job.status, "error": job.error})
        finally:
            # Signal stream end
            await job.events.put({"type": "eof"})

    async def stream(self, jid: str):
        job = self._jobs[jid]
        while True:
            msg = await job.events.get()
            yield msg
            if msg.get("type") == "eof":
                break

    @staticmethod
    def _set_deterministic(seed: Optional[int], deterministic: bool) -> None:
        try:
            import random

            if seed is not None:
                random.seed(seed)
        except Exception:
            pass
        try:
            import numpy as np  # type: ignore

            if seed is not None:
                np.random.seed(seed)
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
