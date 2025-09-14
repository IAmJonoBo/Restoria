from __future__ import annotations

from typing import Any, Dict

from .base import RestoreResult, Restorer


class RestoreFormerPP(Restorer):
    """Adapter for RestoreFormer++ (placeholder using RestoreFormer engine).

    Uses the existing RestoreFormer engine in the gfpgan package as a stand-in
    until RF++ weights are integrated.
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
            from gfpgan.weights import resolve_model_weight
        except Exception as e:  # pragma: no cover
            raise ImportError("gfpgan package not installed for RestoreFormer backend") from e

        # Use RestoreFormer weights for now
        model_name = "RestoreFormer"
        model_path, sha = resolve_model_weight(model_name, no_download=bool(cfg.get("no_download", False)))
        self._model_path = model_path
        dev = "cuda" if (self._device == "auto" and torch.cuda.is_available()) else self._device.replace("auto", "cpu")
        Engine = get_engine("restoreformer")
        self._restorer = Engine(
            model_path=model_path,
            device=torch.device(dev),
            upscale=int(cfg.get("upscale", 2)),
            bg_upsampler=self._bg,
        )

    def restore(self, image, cfg: Dict[str, Any]) -> RestoreResult:
        if self._restorer is None:
            self.prepare(cfg)
        cropped, faces, restored = self._restorer.enhance(
            image,
            has_aligned=bool(cfg.get("aligned", False)),
            only_center_face=bool(cfg.get("only_center_face", False)),
            paste_back=True,
            weight=float(cfg.get("weight", 0.5)),
        )
        return RestoreResult(
            input_path=cfg.get("input_path"),
            restored_path=None,
            restored_image=restored,
            cropped_faces=[],
            restored_faces=[],
            metrics={"model_path": self._model_path},
        )
