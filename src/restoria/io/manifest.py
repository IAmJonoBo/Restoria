from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunManifest:
    args: Dict[str, Any]
    device: str | None
    results: List[Dict[str, Any]]
    metrics_file: Optional[str] = None
    env: Dict[str, Any] = field(default_factory=dict)


def write_manifest(path: str, man: RunManifest) -> None:
    try:
        import json
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "args": man.args,
                    "device": man.device,
                    "results": man.results,
                    "metrics_file": man.metrics_file,
                    "env": man.env,
                },
                f,
                indent=2,
            )
    except Exception:
        pass
