from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .base import RestoreResult, Restorer


def _safe_norm_weights(ws: Sequence[float]) -> List[float]:
    try:
        wsum = float(sum(ws))
        if wsum <= 0:
            return [1.0 / max(1, len(ws)) for _ in ws]
        return [float(w) / wsum for w in ws]
    except Exception:
        n = max(1, len(ws))
        return [1.0 / n for _ in ws]


@dataclass
class _BackendSpec:
    name: str
    weight: float


class EnsembleRestorer(Restorer):
    """Blend outputs from multiple existing backends.

    Notes:
    - Backends are lazy-instantiated on first prepare().
    - If none are available, returns input image unchanged.
    - Blending uses simple weighted average in BGR space.
    - This is opt-in via --backend ensemble and extra flags in CLI.
    """

    def __init__(self, device: str = "auto", bg_upsampler=None) -> None:
        self._device = device
        self._bg = bg_upsampler
        self._inner: List[Tuple[Restorer, float, str]] = []  # (restorer, weight, name)

    def prepare(self, cfg: Dict[str, Any]) -> None:
        from gfpp.core.registry import get as get_backend  # lazy

        names: List[str] = []
        weights: List[float] = []
        try:
            raw_names = cfg.get("ensemble_backends", [])
            if isinstance(raw_names, str):
                names = [s.strip() for s in raw_names.split(",") if s.strip()]
            elif isinstance(raw_names, (list, tuple)):
                names = [str(x) for x in raw_names if str(x)]
        except Exception:
            names = []
        try:
            raw_w = cfg.get("ensemble_weights", [])
            if isinstance(raw_w, str):
                weights = [float(s.strip()) for s in raw_w.split(",") if s.strip()]
            elif isinstance(raw_w, (list, tuple)):
                weights = [float(x) for x in raw_w]
        except Exception:
            weights = []
        if not names:
            # default to a sensible pair if user didn't provide; but remain opt-in
            names = ["gfpgan", "codeformer"]
        if not weights or len(weights) != len(names):
            weights = [1.0 for _ in names]
        weights = _safe_norm_weights(weights)
        self._inner = []
        for n, w in zip(names, weights):
            try:
                backend_cls = get_backend(n)
                rest = backend_cls(device=self._device, bg_upsampler=self._bg)
                # do not prepare now; defer to restore()
                self._inner.append((rest, float(w), n))
            except Exception:
                # skip unavailable backend silently
                continue

    def restore(self, image: Any, cfg: Dict[str, Any]) -> RestoreResult:
        if not self._inner:
            # lazy prepare if needed
            self.prepare(cfg)
        if not self._inner:
            # Nothing available → passthrough
            return RestoreResult(
                input_path=cfg.get("input_path"),
                restored_path=None,
                restored_image=image,
                cropped_faces=[],
                restored_faces=[],
                metrics={"ensemble": {"sources": [], "weights": []}},
            )
        outs: List[Tuple[Any, float, str]] = []
        for rest, w, name in self._inner:
            try:
                r = rest.restore(image, cfg)
                if r and r.restored_image is not None:
                    outs.append((r.restored_image, w, name))
            except Exception:
                continue
        if not outs:
            return RestoreResult(
                input_path=cfg.get("input_path"),
                restored_path=None,
                restored_image=image,
                cropped_faces=[],
                restored_faces=[],
                metrics={"ensemble": {"sources": [], "weights": []}},
            )
        # Blend images by normalized weights
        try:
            import numpy as np  # type: ignore

            acc: Any = None
            wsum = 0.0
            srcs = []
            ws = []
            for img, w, name in outs:
                arr = np.asarray(img, dtype=np.float32)
                if acc is None:
                    acc = np.zeros_like(arr, dtype=np.float32)
                acc += float(w) * arr
                wsum += float(w)
                srcs.append(name)
                ws.append(float(w))
            if wsum <= 0:
                blend = outs[0][0]
            else:
                acc = (acc / float(wsum)).astype(np.float32)
                blend = np.clip(acc, 0, 255).astype("uint8")
        except Exception:
            # Fallback: return first
            blend = outs[0][0]
            srcs = [outs[0][2]]
            ws = [outs[0][1]]
        return RestoreResult(
            input_path=cfg.get("input_path"),
            restored_path=None,
            restored_image=blend,
            cropped_faces=[],
            restored_faces=[],
            metrics={"ensemble": {"sources": srcs, "weights": ws}},
        )
