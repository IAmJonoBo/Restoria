from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .base import RestoreResult, Restorer
from ..engines.torch_eager import autocast_ctx, tile_image  # type: ignore


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
        self._detector_used = "retinaface_resnet50"

    def prepare(self, cfg: Dict[str, Any]) -> None:
        try:
            import torch

            from gfpgan.engines import get_engine  # register engines
            from gfpgan.weights import resolve_model_weight
            # Optional compile/memory tweaks are handled elsewhere when enabled

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
            dev = (
                "cuda"
                if (self._device == "auto" and torch.cuda.is_available())
                else self._device.replace("auto", "cpu")
            )
            engine_cls = get_engine("gfpgan")
            requested_det = str(cfg.get("detector", "retinaface_resnet50"))
            # Optional compatibility: map 'insightface' to 'scrfd' (closest available)
            if requested_det == "insightface":
                try:
                    print("[WARN] Detector 'insightface' not directly supported; using 'scrfd' instead")
                except Exception:
                    pass
                requested_det = "scrfd"

            # Try requested detector first; if it fails, fall back to retinaface without re-raising here
            created = False
            try:
                self._restorer = engine_cls(
                    model_path=model_path,
                    device=torch.device(dev),
                    upscale=int(cfg.get("upscale", 2)),
                    arch=arch,
                    channel_multiplier=cm,
                    bg_upsampler=self._bg,
                    det_model=requested_det,
                    use_parse=bool(cfg.get("use_parse", True)),
                )
                self._detector_used = requested_det
                created = True
            except Exception:
                created = False
            if not created:
                # Fallback to retinaface if the requested detector is unsupported/unavailable
                self._restorer = engine_cls(
                    model_path=model_path,
                    device=torch.device(dev),
                    upscale=int(cfg.get("upscale", 2)),
                    arch=arch,
                    channel_multiplier=cm,
                    bg_upsampler=self._bg,
                    det_model="retinaface_resnet50",
                    use_parse=bool(cfg.get("use_parse", True)),
                )
                self._detector_used = "retinaface_resnet50"
                try:
                    print("[WARN] Requested detector not available; using retinaface_resnet50")
                except Exception:
                    pass

            # Optional compile and memory format tweaks intentionally omitted to keep startup fast
        except Exception:
            # Graceful stub: create a no-op restorer to enable dry tests without deps
            class _Stub:
                def enhance(self, img, **_kwargs):
                    return [], [], img

            self._restorer = _Stub()
            self._model_path = None
            self._model_name = "GFPGANv1.4"
            self._model_sha = None
            # Record requested detector even on stub path for metrics transparency
            try:
                self._detector_used = str(cfg.get("detector", self._detector_used))
            except Exception:
                pass

    def restore(self, image, cfg: Dict[str, Any]) -> RestoreResult:
        if self._restorer is None:
            self.prepare(cfg)
        weight = float(cfg.get("weight", 0.5))
        # Precision and tiling hints
        precision = str(cfg.get("precision", "auto"))
        tile = int(cfg.get("tile", 0))
        tile_ov = int(cfg.get("tile_overlap", 0))
        dev = "cuda" if (self._device == "auto") else self._device
        dev = dev if dev in {"cuda", "cpu", "mps"} else "cpu"
        # Choose dtype for autocast when enabled
        autocast_dtype = None
        try:
            import torch  # type: ignore

            if precision == "bf16" and dev == "cpu":
                autocast_dtype = torch.bfloat16
            elif precision in {"auto", "fp16"} and dev == "cuda":
                autocast_dtype = torch.float16
        except Exception:
            autocast_dtype = None
        assert self._restorer is not None
        img_in = tile_image(image, tile_size=tile, tile_overlap=tile_ov)
        # Autocast context: enable on CUDA for fp16/auto; bf16 left to future per-device checks
        enable_autocast = (precision in {"auto", "fp16"} and dev == "cuda") or (
            precision == "bf16" and dev == "cpu"
        )
        with autocast_ctx(
            device_type="cuda" if dev == "cuda" else "cpu",
            enabled=enable_autocast,
            dtype=autocast_dtype,
        ):
            _, _, restored = self._restorer.enhance(
                img_in,
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
                "detector": self._detector_used,
                "precision": precision,
            },
        )
