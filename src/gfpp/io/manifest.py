from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _sha256(path: str) -> Optional[str]:
    try:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def collect_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        info["torch"] = getattr(torch, "__version__", None)
        info["cuda"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        if info["cuda"]:
            try:
                info["cuda_device"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception:
        info["torch"] = None
        info["cuda"] = False
    try:
        import subprocess

        info["git"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        info["git"] = None
    return info


@dataclass
class RunManifest:
    args: Dict[str, Any]
    device: str
    started_at: float = field(default_factory=lambda: time.time())
    ended_at: Optional[float] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    models: List[Dict[str, Any]] = field(default_factory=list)
    metrics_file: Optional[str] = None
    env: Dict[str, Any] = field(default_factory=collect_env)


def write_manifest(path: str, man: RunManifest) -> None:
    data = asdict(man)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
