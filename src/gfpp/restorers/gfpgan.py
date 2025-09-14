from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import RestoreResult, Restorer


@dataclass
class _Cfg:
    upscale: int = 2
    arch: str = "clean"
    channel_multiplier: int = 2
    detector: str = "retinaface_resnet50"
    use_parse: bool = True
    weight: float = 0.5


class GFPGANRestorer(Restorer):
    """Adapter around existing GFPGANer from the gfpgan package.

    Keeps imports lazy and works without adding new hard dependencies.
    """

    def __init__(self, device: str = "auto", bg_upsampler=None, compile_mode: str = "none") -> None:
        self._device = device
        self._bg = bg_upsampler
        self._restorer = None
        self._model_path = None
        self._model_name = None
        self._model_sha = None
        self._compile_mode = compile_mode

    def prepare(self, cfg: Dict[str, Any]) -> None:
        import os

        import torch

        from gfpgan.engines import get_engine  # register engines
        from gfpgan.weights import resolve_model_weight
        from src.gfpp.engines.torch_compile import compile_module  # type: ignore
        from src.gfpp.engines.torch_eager import to_channels_last  # type: ignore

        version = str(cfg.get("version", "1.4"))
        if version == "1":
            arch, cm, model_name = "original", 1, "GFPGANv1"
        elif version == "1.2":
            arch, cm, model_name = "clean", 2, "GFPGANCleanv1-NoCE-C2"
        elif version == "1.3":
            arch, cm, model_name = "clean", 2, "GFPGANv1.3"
        else:
            arch, cm, model_name = "clean", 2, "GFPGANv1.4"

        model_path, sha = resolve_model_weight(model_name, no_download=bool(cfg.get("no_download", False)))
        self._model_path = model_path
        self._model_name = model_name
        self._model_sha = sha
        dev = "cuda" if (self._device == "auto" and torch.cuda.is_available()) else self._device.replace("auto", "cpu")
        Engine = get_engine("gfpgan")
        self._restorer = Engine(
            model_path=model_path,
            device=torch.device(dev),
            upscale=int(cfg.get("upscale", 2)),
            arch=arch,
            channel_multiplier=cm,
            bg_upsampler=self._bg,
            det_model=str(cfg.get("detector", "retinaface_resnet50")),
            use_parse=bool(cfg.get("use_parse", True)),
        )
        # Optional compile and memory format tweaks
        try:
            if getattr(self._restorer, "gfpgan", None) is not None:
                self._restorer.gfpgan = to_channels_last(self._restorer.gfpgan)
                self._restorer.gfpgan = compile_module(self._restorer.gfpgan, mode=self._compile_mode)
        except Exception:
            pass

    def restore(self, image, cfg: Dict[str, Any]) -> RestoreResult:
        import numpy as np

        if self._restorer is None:
            self.prepare(cfg)
        weight = float(cfg.get("weight", 0.5))
        cropped, faces, restored = self._restorer.enhance(
            image,
            has_aligned=bool(cfg.get("aligned", False)),
            only_center_face=bool(cfg.get("only_center_face", False)),
            paste_back=True,
            weight=weight,
            eye_dist_threshold=int(cfg.get("eye_dist_threshold", 5)),
        )
        return RestoreResult(
            input_path=cfg.get("input_path"),
            restored_path=None,
            restored_image=restored,
            cropped_faces=[str(x) for x in []],
            restored_faces=[str(x) for x in []],
            metrics={
                "model_name": self._model_name,
                "model_path": self._model_path,
                "model_sha256": self._model_sha,
            },
        )
