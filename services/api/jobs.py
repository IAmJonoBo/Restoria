from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .schemas import JobSpec


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
                    await job.events.put(
                        {
                            "type": "image",
                            "input": pth,
                            "output": out_img,
                            "metrics": metrics,
                            "progress": job.progress,
                        }
                    )
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
                    # Auto backend per image
                    if spec.auto_backend:
                        try:
                            from gfpgan.auto.engine_selector import select_engine_for_image  # type: ignore

                            bname = select_engine_for_image(pth).engine
                        except Exception:
                            bname = spec.backend
                        if bname != spec.backend:
                            if bname == "gfpgan":
                                rest = GFPGANRestorer(device="auto", bg_upsampler=bg)
                            elif bname == "codeformer":
                                rest = CodeFormerRestorer(device="auto", bg_upsampler=bg)
                            elif bname in {"restoreformer", "restoreformerpp"}:
                                rest = RestoreFormerPP(device="auto", bg_upsampler=bg)

                    # Pass through model_path_onnx when present
                    if isinstance(rest, ORTGFPGANRestorer) and spec.model_path_onnx:
                        cfg["model_path_onnx"] = spec.model_path_onnx

                    # Optional optimize: try several weights and pick best by metric
                    chosen_weight = cfg.get("weight")
                    res = None
                    if spec.optimize:
                        try:
                            cand = [
                                min(max(float(x.strip()), 0.0), 1.0) for x in spec.weights_cand.split(",") if x.strip()
                            ]
                        except Exception:
                            cand = [0.3, 0.5, 0.7]
                        import cv2
                        import tempfile

                        best_score = None
                        best_res = None
                        best_w = None
                        for w in cand:
                            cfg["weight"] = w
                            r_try = rest.restore(img, cfg)
                            score = None
                            if arc and arc.available():
                                td = tempfile.mkdtemp()
                                a = os.path.join(td, "in.png")
                                b = os.path.join(td, "out.png")
                                cv2.imwrite(a, img)
                                cv2.imwrite(
                                    b,
                                    r_try.restored_image if (r_try and r_try.restored_image is not None) else img,
                                )
                                s = arc.cosine_from_paths(a, b)
                                score = s if s is not None else None
                            if score is None and lpips and lpips.available():
                                td = tempfile.mkdtemp()
                                a = os.path.join(td, "in.png")
                                b = os.path.join(td, "out.png")
                                cv2.imwrite(a, img)
                                cv2.imwrite(
                                    b,
                                    r_try.restored_image if (r_try and r_try.restored_image is not None) else img,
                                )
                                d = lpips.distance_from_paths(a, b)
                                score = -d if isinstance(d, float) else None
                            if score is None:
                                score = -abs(w - float(preset_weight))
                            if best_score is None or score > best_score:
                                best_score, best_res, best_w = score, r_try, w
                        res = best_res
                        chosen_weight = best_w
                        cfg["weight"] = chosen_weight
                    else:
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
                    # Identity lock retry if below threshold
                    if spec.identity_lock and arc and arc.available():
                        import cv2
                        import tempfile

                        td = tempfile.mkdtemp()
                        a = os.path.join(td, "in.png")
                        b = os.path.join(td, "out.png")
                        cv2.imwrite(a, img)
                        cv2.imwrite(b, res.restored_image if (res and res.restored_image is not None) else img)
                        s0 = arc.cosine_from_paths(a, b)
                        if s0 is not None and s0 < float(spec.identity_threshold):
                            cfg_strict = dict(cfg)
                            cfg_strict["weight"] = max(0.2, float(cfg.get("weight", preset_weight)) - 0.2)
                            r2 = rest.restore(img, cfg_strict)
                            cv2.imwrite(b, r2.restored_image if (r2 and r2.restored_image is not None) else img)
                            s1 = arc.cosine_from_paths(a, b)
                            if s1 is not None and s1 > (s0 or 0):
                                res = r2
                                save_image(
                                    out_img, res.restored_image if res and res.restored_image is not None else img
                                )
                                metrics["identity_retry"] = True
                    count += 1
                    job.result_count = count
                    job.progress = count / n
                    results_rec.append({"input": pth, "restored_img": out_img, "metrics": metrics})
                    await job.events.put(
                        {"type": "image", "input": pth, "output": out_img, "metrics": metrics, "progress": job.progress}
                    )

            # Write manifest.json and optional metrics/report for reproducibility
            try:
                out_dir = job.results_path or spec.output
                man = RunManifest(args=asdict(spec), device="auto", results=results_rec)
                write_manifest(os.path.join(out_dir, "manifest.json"), man)
                await job.events.put({"type": "manifest", "path": os.path.join(out_dir, "manifest.json")})
            except Exception:
                pass
            # metrics.json
            try:
                import json as _json

                with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                    _json.dump({"metrics": results_rec}, f, indent=2)
            except Exception:
                pass
            # HTML report
            try:
                from src.gfpp.reports.html import write_html_report  # type: ignore

                write_html_report(out_dir, results_rec, path=os.path.join(out_dir, "report.html"))
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
