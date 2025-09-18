from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List
import importlib
try:
    # Python 3.10+
    from importlib.metadata import entry_points  # type: ignore
except Exception:
    entry_points = None  # type: ignore


@dataclass(frozen=True)
class BackendInfo:
    spec: str
    latency: str = "medium"
    devices: List[str] = field(default_factory=lambda: ["auto"])
    license: str = "Apache-2.0"
    noncommercial: bool = False
    experimental: bool = False
    description: str = ""


_MAP: Dict[str, BackendInfo] = {
    "gfpgan": BackendInfo(
        spec="gfpp.restorers.gfpgan:GFPGANRestorer",
        latency="medium",
        devices=["cpu", "cuda", "mps"],
        description="GFPGAN baseline restorer",
    ),
    "gfpgan-ort": BackendInfo(
        spec="gfpp.restorers.gfpgan_ort:ORTGFPGANRestorer",
        latency="fast",
        devices=["cuda", "cpu"],
        description="GFPGAN ONNX Runtime backend",
    ),
    "codeformer": BackendInfo(
        spec="gfpp.restorers.codeformer:CodeFormerRestorer",
        latency="slow",
        devices=["cuda", "cpu"],
        license="NTU S-Lab 1.0",
        noncommercial=True,
        description="CodeFormer high-fidelity restorer",
    ),
    "restoreformerpp": BackendInfo(
        spec="gfpp.restorers.restoreformerpp:RestoreFormerPP",
        latency="medium",
        devices=["cuda"],
        description="RestoreFormer++ restorer",
    ),
    "diffbir": BackendInfo(
        spec="gfpp.restorers.diffbir:DiffBIRRestorer",
        latency="slow",
        devices=["cuda"],
        experimental=True,
        description="DiffBIR experimental backend",
    ),
    "hypir": BackendInfo(
        spec="gfpp.restorers.hypir:HYPIRRestorer",
        latency="slow",
        devices=["cuda"],
        experimental=True,
        description="HYPIR experimental backend",
    ),
}

_health_cache: Dict[str, bool] = {}


def _import_symbol(spec: str):
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def _plugins() -> Dict[str, BackendInfo]:
    out: Dict[str, BackendInfo] = {}
    if entry_points is None:
        return out
    try:
        eps = entry_points()
        # New API returns dict-like; older returns list -> handle both
        group = None
        if hasattr(eps, "select"):
            group = eps.select(group="gfpp.restorers")  # type: ignore[attr-defined]
        else:
            group = [e for e in eps if getattr(e, "group", "") == "gfpp.restorers"]  # type: ignore[assignment]
        for ep in group or []:
            try:
                name = getattr(ep, "name", None)
                spec = getattr(ep, "value", None) or getattr(ep, "module", None)
                if name and spec and isinstance(name, str) and isinstance(spec, str):
                    out[name] = BackendInfo(spec=spec, description="External plugin", devices=["auto"])
            except Exception:
                continue
    except Exception:
        return out
    return out


def get(name: str):
    # Merge built-in map with any discovered plugins (plugins win on conflicts)
    all_map = dict(_MAP)
    try:
        all_map.update(_plugins())
    except Exception:
        pass
    info = all_map.get(name)
    if not info:
        raise KeyError(name)
    return _import_symbol(info.spec)


def _check_available(name: str, info: BackendInfo) -> bool:
    if name in _health_cache:
        return _health_cache[name]
    try:
        _import_symbol(info.spec)
        _health_cache[name] = True
    except Exception:
        _health_cache[name] = False
    return _health_cache[name]


def list_backends(include_experimental: bool = False) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    all_map = dict(_MAP)
    try:
        all_map.update(_plugins())
    except Exception:
        pass
    for name, info in all_map.items():
        if info.experimental and not include_experimental:
            continue
        available = _check_available(name, info)
        metadata = asdict(info)
        out[name] = {"available": available, "metadata": metadata}
    return out
