from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, Optional


def collect_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        if info["cuda_available"]:
            try:
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["cuda_total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            except Exception:
                pass
    except Exception:
        info["torch_version"] = None
        info["cuda_available"] = False
    try:
        info["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        info["git_commit"] = None
    return info


def build_manifest(
    *,
    args: Dict[str, Any],
    device: str,
    model_name: str,
    model_path: str,
    model_sha256: Optional[str],
    results: Any,
    metrics_path: Optional[str] = None,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "meta": {
            "device": device,
            "model_name": model_name,
            "model_path": model_path,
            "model_sha256": model_sha256,
            **collect_env(),
        },
        "args": args,
        "results": results,
    }
    if metrics_path:
        data["metrics_file"] = metrics_path
    return data


def write_manifest(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def set_deterministic(seed: Optional[int] = None, deterministic_cuda: bool = False) -> None:
    """Set random seeds and deterministic flags where available."""
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
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass


__all__ = ["collect_env", "build_manifest", "write_manifest", "set_deterministic"]
