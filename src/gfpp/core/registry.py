from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

# Minimal, pluggable backend registry with lazy imports


@dataclass
class BackendInfo:
    name: str
    loader: Callable[[], Any]
    experimental: bool = False
    available: Optional[bool] = None  # cached availability


_REGISTRY: Dict[str, BackendInfo] = {}


def _try_import(path: str) -> Optional[Any]:
    try:
        mod_path, cls_name = path.rsplit(":", 1)
        mod = __import__(mod_path, fromlist=[cls_name])
        return getattr(mod, cls_name)
    except Exception:
        return None


def register(name: str, target: str, *, experimental: bool = False) -> None:
    def _load():
        cls = _try_import(target)
        if cls is None:
            raise ImportError(f"Backend '{name}' unavailable: could not import {target}")
        return cls

    _REGISTRY[name] = BackendInfo(name=name, loader=_load, experimental=experimental)


# Built-ins: wired to existing adapters
register("gfpgan", "gfpp.restorers.gfpgan:GFPGANRestorer")
register("gfpgan-ort", "gfpp.restorers.gfpgan_ort:ORTGFPGANRestorer", experimental=True)
register("codeformer", "gfpp.restorers.codeformer:CodeFormerRestorer")
register("restoreformerpp", "gfpp.restorers.restoreformerpp:RestoreFormerPP")
register("diffbir", "gfpp.restorers.diffbir:DiffBIRRestorer", experimental=True)
register("hypir", "gfpp.restorers.hypir:HYPIRRestorer", experimental=True)
register("ensemble", "gfpp.restorers.ensemble:EnsembleRestorer")
register("guided", "gfpp.restorers.guided:GuidedRestorer", experimental=True)


def list_backends(include_experimental: bool = False) -> Dict[str, bool]:
    """Return mapping of backend name to availability.

    Availability is tested by trying to resolve the loader target once.
    """
    out: Dict[str, bool] = {}
    for name, info in _REGISTRY.items():
        if info.experimental and not include_experimental:
            continue
        if info.available is None:
            cls = None
            try:
                cls = info.loader()
            except Exception:
                cls = None
            info.available = cls is not None
        out[name] = bool(info.available)
    return out


essential_aliases = {
    "restoreformer": "restoreformerpp",
}


def get(name: str):
    """Return the backend class for name or raise KeyError.

    Accepts a few aliases.
    """
    name = essential_aliases.get(name, name)
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend: {name}")
    info = _REGISTRY[name]
    cls = info.loader()
    return cls
