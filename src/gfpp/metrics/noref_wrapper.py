from __future__ import annotations

from typing import Dict

from .norefs import niqe as _niqe, brisque as _brisque


class NoRefQuality:
    """Aggregate no-reference quality metrics with graceful fallbacks.

    Provides a unified interface so callers can do:
        q = NoRefQuality(); scores = q.score(path)
    where `scores` is a dict possibly containing 'niqe' and/or 'brisque'.

    Rationale: Simplifies orchestration & manifest writing by avoiding
    per-metric branching. Missing metrics are silently omitted.
    """

    def available(self) -> bool:  # at least one metric import path works
        return (_niqe.__code__ is not None) or (_brisque.__code__ is not None)  # type: ignore[attr-defined]

    def score(self, path: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            n = _niqe(path)
            if n is not None:
                out["niqe"] = float(n)
        except Exception:
            pass
        try:
            b = _brisque(path)
            if b is not None:
                out["brisque"] = float(b)
        except Exception:
            pass
        return out
