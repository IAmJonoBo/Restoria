from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Dict
import importlib


_MAP: Dict[str, str] = {
    "gfpgan": "gfpp.restorers.gfpgan:GFPGANRestorer",
    "gfpgan-ort": "gfpp.restorers.gfpgan_ort:ORTGFPGANRestorer",
    "codeformer": "gfpp.restorers.codeformer:CodeFormerRestorer",
    "restoreformerpp": "gfpp.restorers.restoreformerpp:RestoreFormerPP",
    # Experimental placeholders
    "diffbir": "gfpp.restorers.diffbir:DiffBIRRestorer",
    "hypir": "gfpp.restorers.hypir:HYPIRRestorer",
}


def _import_symbol(spec: str):
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def get(name: str):
    spec = _MAP.get(name)
    if not spec:
        raise KeyError(name)
    return _import_symbol(spec)


def list_backends(include_experimental: bool = False) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for k, v in _MAP.items():
        if not include_experimental and k in {"diffbir", "hypir"}:
            continue
        ok = True
        try:
            _import_symbol(v)
        except Exception:
            ok = False
        out[k] = ok
    return out
