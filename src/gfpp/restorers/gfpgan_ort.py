from __future__ import annotations

from typing import Any, Dict, Optional

from .base import RestoreResult, Restorer


class ORTGFPGANRestorer(Restorer):
    """ONNX Runtime path for GFPGAN (scaffold with graceful fallback).

    - Detects available execution providers and logs selection
    - If ORT session cannot be created (no model graph or ORT missing), falls back to Torch GFPGANRestorer
    - API shape matches other restorers
    """

    def __init__(self, device: str = "auto", bg_upsampler=None) -> None:
        self._device = device
        self._bg = bg_upsampler
        self._ort = None
        self._ort_info: Dict[str, Any] = {}
        self._fallback = None
        self._ort_ready = False

    def prepare(self, cfg: Dict[str, Any]) -> None:
        from time import perf_counter

        from src.gfpp.engines.onnxruntime import available_eps, create_session, session_info  # type: ignore
        from .gfpgan import GFPGANRestorer

        # Users may pass an ONNX model path via cfg["model_path_onnx"]. If missing, fallback.
        onnx_path: Optional[str] = cfg.get("model_path_onnx")
        eps = available_eps()
        self._ort_info["available_eps"] = eps

        if onnx_path:
            t0 = perf_counter()
            sess = create_session(onnx_path)
            init_sec = perf_counter() - t0
            if sess is not None:
                self._ort = sess
                self._ort_info.update(session_info(sess))
                self._ort_info["init_sec"] = round(init_sec, 4)
                self._ort_ready = True
        if self._ort is None:
            # Prepare fallback Torch restorer
            self._fallback = GFPGANRestorer(device=self._device, bg_upsampler=self._bg, compile_mode=cfg.get("compile", "none"))
            self._fallback.prepare(cfg)

    def restore(self, image, cfg: Dict[str, Any]) -> RestoreResult:
        # If ORT session available, this is where the actual ONNX inference would happen.
        # Since exporting GFPGAN to ONNX is out-of-scope for this scaffold, we always fallback for now.
        if self._ort is None:
            if self._fallback is None:
                self.prepare(cfg)
            res = self._fallback.restore(image, cfg)
            # Attach ORT metadata for transparency
            res.metrics.update({
                "backend": "torch-fallback",
                "ort_available_eps": ",".join(self._ort_info.get("available_eps", []) or []),
                "ort_provider": self._ort_info.get("providers"),
                "ort_init_sec": self._ort_info.get("init_sec"),
            })
            return res

        # Placeholder path: ORT session exists, but generator graph I/O mapping is not implemented yet.
        # Return fallback image if available; otherwise, pass-thru image. Still record ORT provider info.
        if self._fallback is None:
            self._fallback = GFPGANRestorer(device=self._device, bg_upsampler=self._bg, compile_mode=cfg.get("compile", "none"))
            self._fallback.prepare(cfg)
        res = self._fallback.restore(image, cfg)
        res.metrics.update({
            "backend": "onnxruntime+torch-fallback",
            "ort_provider": self._ort_info.get("providers"),
            "ort_init_sec": self._ort_info.get("init_sec"),
        })
        return res
