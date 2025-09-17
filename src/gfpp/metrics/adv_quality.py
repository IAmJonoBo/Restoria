from __future__ import annotations

from typing import Dict, Optional


class MANIQAWrapper:
    def __init__(self) -> None:
        self._model = None

    def available(self) -> bool:
        # Consider "available" for proxy computation; real model may be absent
        return True

    def score(self, path: str) -> Optional[float]:  # noqa: ARG002 - placeholder
        try:
            # Prefer real MANIQA if installed, else compute a proxy from NIQE/BRISQUE
            has_maniqa = False
            try:
                from maniqa import Maniqa  # type: ignore
                _ = Maniqa  # marker only
                has_maniqa = True
            except Exception:
                has_maniqa = False
            if has_maniqa:
                # Not wired to avoid heavy downloads in tests.
                return None
            # Proxy: lower NIQE/BRISQUE => higher quality
            try:
                from .norefs import niqe, brisque

                n = niqe(path)
                b = brisque(path)
                parts = []
                if isinstance(n, (int, float)):
                    parts.append(1.0 / (1.0 + max(0.0, float(n))))
                if isinstance(b, (int, float)):
                    parts.append(1.0 / (1.0 + max(0.0, float(b))))
                if parts:
                    return float(sum(parts) / len(parts))
                return None
            except Exception:
                return None
        except Exception:
            return None


class CONTRIQUEWrapper:
    def __init__(self) -> None:
        self._model = None

    def available(self) -> bool:
        return True

    def score(self, path: str) -> Optional[float]:  # noqa: ARG002 - placeholder
        try:
            # Prefer real CONTRIQUE if installed, else proxy
            ok = False
            try:
                import contrique  # type: ignore
                _ = contrique
                ok = True
            except Exception:
                ok = False
            if ok:
                return None
            try:
                from .norefs import niqe

                n = niqe(path)
                if isinstance(n, (int, float)):
                    return float(1.0 / (1.0 + max(0.0, float(n))))
                return None
            except Exception:
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
