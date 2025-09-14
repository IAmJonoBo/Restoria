from __future__ import annotations

from typing import Any, Dict

from .base import RestoreResult, Restorer


class CodeFormerRestorer(Restorer):
    """Adapter for CodeFormer using existing engine in gfpgan package.

    Optional dependency; if backend missing, raises a clear ImportError.
    """

    def __init__(self, device: str = "auto", bg_upsampler=None) -> None:
        self._device = device
        self._bg = bg_upsampler
        self._restorer = None
        self._model_path = None

    def prepare(self, cfg: Dict[str, Any]) -> None:
        import torch

        try:
            from gfpgan.engines import get_engine
        except Exception as e:  # pragma: no cover
            raise ImportError("gfpgan package not installed for CodeFormer backend") from e

        # CodeFormer weights are resolved by users; we default to upstream URL
        model_path = cfg.get(
            "model_path",
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        )
        self._model_path = model_path
        dev = "cuda" if (self._device == "auto" and torch.cuda.is_available()) else self._device.replace("auto", "cpu")
        Engine = get_engine("codeformer")
        self._restorer = Engine(
            model_path=model_path,
            device=torch.device(dev),
            upscale=int(cfg.get("upscale", 2)),
            bg_upsampler=self._bg,
        )

    def restore(self, image, cfg: Dict[str, Any]) -> RestoreResult:
        if self._restorer is None:
            self.prepare(cfg)
        weight = float(cfg.get("weight", 0.5))
        cropped, faces, restored = self._restorer.enhance(
            image,
            has_aligned=bool(cfg.get("aligned", False)),
            only_center_face=bool(cfg.get("only_center_face", False)),
            paste_back=True,
            weight=weight,
        )
        return RestoreResult(
            input_path=cfg.get("input_path"),
            restored_path=None,
            restored_image=restored,
            cropped_faces=[],
            restored_faces=[],
            metrics={"model_path": self._model_path},
        )
