from __future__ import annotations

import os
import sys
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="GFPGAN API", version="0.1")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "True", "yes"}


@app.get("/healthz")
def healthz():
    # Minimal, dependency-light report
    info = {
        "status": "ok",
        "python": sys.version.split()[0],
    }
    try:
        import torch  # type: ignore

        info["torch"] = getattr(torch, "__version__", None)
        info["cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        info["torch"] = None
        info["cuda_available"] = False
    return JSONResponse(info)


@app.post("/restore")
async def restore(
    version: str = "1.4",
    upscale: int = 2,
    backend: str = "gfpgan",
    device: str = "auto",
    dry_run: bool = True,
    files: List[UploadFile] = File(...),
):
    # Smoke-friendly default: dry_run True. In that mode, only echo back metadata.
    if dry_run or _env_flag("NB_CI_SMOKE"):
        # Return a manifest-shaped response without doing any work
        names = [f.filename for f in files]
        results = [
            {"input": n, "restored_imgs": [], "restored_faces": [], "cropped_faces": [], "weights": [0.5]}
            for n in names
        ]
        return JSONResponse(
            {
                "accepted": len(names),
                "params": {
                    "version": version,
                    "upscale": upscale,
                    "backend": backend,
                    "device": device,
                },
                "results": results,
                "dry_run": True,
            }
        )

    # Process a single in-memory image (minimal path). Can be extended to
    # batch + manifest-like outputs.
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        # Read first image
        if not files:
            return JSONResponse({"error": "no files uploaded"}, status_code=400)
        data = await files[0].read()
        img_arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # Map version -> model_name/arch/cm (reuse a subset from CLI)
        if backend == "gfpgan":
            if version == "1":
                arch, cm, model_name = "original", 1, "GFPGANv1"
            elif version == "1.2":
                arch, cm, model_name = "clean", 2, "GFPGANCleanv1-NoCE-C2"
            elif version == "1.3":
                arch, cm, model_name = "clean", 2, "GFPGANv1.3"
            else:
                arch, cm, model_name = "clean", 2, "GFPGANv1.4"
            # Resolve weight path
            import torch

            from gfpgan.engines import get_engine
            from gfpgan.weights import resolve_model_weight

            model_path, _ = resolve_model_weight(model_name, no_download=False)
            Engine = get_engine("gfpgan")
            restorer = Engine(
                model_path=model_path,
                device=torch.device("cuda" if device == "cuda" else "cpu"),
                upscale=upscale,
                arch=arch,
                channel_multiplier=cm,
                bg_upsampler=None,
                det_model="retinaface_resnet50",
                use_parse=True,
            )
            _, _, restored = restorer.enhance(img, has_aligned=False, paste_back=True, weight=0.5)
            ok = restored is not None
        elif backend == "restoreformer":
            import torch

            from gfpgan.engines import get_engine
            from gfpgan.weights import resolve_model_weight

            Engine = get_engine("restoreformer")
            model_name = "RestoreFormer"
            model_path, _ = resolve_model_weight(model_name, no_download=False)
            restorer = Engine(
                model_path=model_path,
                device=torch.device("cuda" if device == "cuda" else "cpu"),
                upscale=upscale,
                bg_upsampler=None,
            )
            _, _, restored = restorer.enhance(img, has_aligned=False, paste_back=True, weight=0.5)
            ok = restored is not None
        elif backend == "codeformer":
            import torch

            from gfpgan.engines import get_engine

            Engine = get_engine("codeformer")
            model_path = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
            restorer = Engine(
                model_path=model_path,
                device=torch.device("cuda" if device == "cuda" else "cpu"),
                upscale=upscale,
                bg_upsampler=None,
            )
            _, _, restored = restorer.enhance(img, has_aligned=False, paste_back=True, weight=0.5)
            ok = restored is not None
        else:
            return JSONResponse({"error": f"Unknown backend: {backend}"}, status_code=400)

        return JSONResponse({"ok": ok})
    except Exception as e:  # pragma: no cover (requires runtime deps)
        return JSONResponse({"error": str(e)}, status_code=500)


def main():  # pragma: no cover - convenience entrypoint
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")  # Default to localhost for security
    uvicorn.run("gfpgan.api.server:app", host=host, port=port, reload=False)
