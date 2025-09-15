from __future__ import annotations

from typing import Dict, Optional

from gfpp.metrics.norefs import brisque as _brisque, niqe as _niqe


def probe_quality(path: str) -> Optional[Dict[str, float]]:
    """Return best-effort NIQE/BRISQUE metrics.

    Returns None if neither is available.
    """
    n = _niqe(path)
    b = _brisque(path)
    if n is None and b is None:
        return None
    out: Dict[str, float] = {}
    if n is not None:
        out["niqe"] = float(n)
    if b is not None:
        out["brisque"] = float(b)
    return out
