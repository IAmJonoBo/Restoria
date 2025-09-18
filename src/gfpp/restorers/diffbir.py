from __future__ import annotations

from typing import Any, Dict

from .base import RestoreResult, Restorer


class DiffBIRRestorer(Restorer):
    """EXPERIMENTAL: DiffBIR adapter with graceful fallback.

    - Tries to import a diffusion-based backend lazily (optional extra)
    - If unavailable, returns the input image unchanged and annotates metrics
    """

    def __init__(self, device: str = "auto", bg_upsampler=None) -> None:
        self._device = device
        self._bg = bg_upsampler
        self._impl = None

    def prepare(self, cfg: Dict[str, Any]) -> None:
        try:
            # Try resolving an implementation if present in the package tree
            from gfpgan.engines import get_engine  # type: ignore
            import torch  # type: ignore

            dev = (
                "cuda"
                if (self._device == "auto" and torch.cuda.is_available())
                else self._device.replace("auto", "cpu")
            )
            engine_cls = get_engine("diffbir")
            self._impl = engine_cls(device=torch.device(dev), bg_upsampler=self._bg)
        except Exception:
            self._impl = None

    def restore(self, image, cfg: Dict[str, Any]) -> RestoreResult:
        if self._impl is None:
            # Lazy prepare if needed
            self.prepare(cfg)
        if self._impl is None:
            # Fallback: return input image and annotate
            return RestoreResult(
                input_path=cfg.get("input_path"),
                restored_path=None,
                restored_image=image,
                cropped_faces=[],
                restored_faces=[],
                metrics={
                    "backend": "diffbir",
                    "fallback": True,
                    "reason": "diffbir engine unavailable",
                },
            )
        try:
            cropped, faces, restored = self._impl.enhance(
                image,
                has_aligned=bool(cfg.get("aligned", False)),
                paste_back=True,
            )
        except Exception:
            restored = image
            cropped, faces = [], []
        return RestoreResult(
            input_path=cfg.get("input_path"),
            restored_path=None,
            restored_image=restored,
            cropped_faces=[str(x) for x in cropped],
            restored_faces=[str(x) for x in faces],
            metrics={"backend": "diffbir", "fallback": self._impl is None},
        )
