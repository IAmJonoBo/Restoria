from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import RunManifest


def write_manifest(path: str, man: RunManifest | dict[str, Any]) -> None:
    payload = man if isinstance(man, dict) else man.model_dump(exclude_none=True)
    try:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass


__all__ = ["RunManifest", "write_manifest"]
