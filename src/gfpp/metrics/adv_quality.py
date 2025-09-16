from __future__ import annotations

from typing import Dict, Optional


class MANIQAWrapper:
    def __init__(self) -> None:
        self._model = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            # placeholder: in real impl, check weights on disk
            return True
        except Exception:
            return False

    def score(self, path: str) -> Optional[float]:  # noqa: ARG002 - placeholder
        try:
            if not self.available():
                return None
            # Placeholder: Return None to indicate unavailable score by default
            return None
        except Exception:
            return None


class CONTRIQUEWrapper:
    def __init__(self) -> None:
        self._model = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def score(self, path: str) -> Optional[float]:  # noqa: ARG002 - placeholder
        try:
            if not self.available():
                return None
            return None
        except Exception:
            return None


def advanced_scores(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        m = MANIQAWrapper()
        s = m.score(path)
        if isinstance(s, (int, float)):
            out["maniqa"] = float(s)
    except Exception:
        pass
    try:
        c = CONTRIQUEWrapper()
        s = c.score(path)
        if isinstance(s, (int, float)):
            out["contrique"] = float(s)
    except Exception:
        pass
    return out
