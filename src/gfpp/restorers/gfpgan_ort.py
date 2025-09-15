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
        from gfpp.engines.onnxruntime import available_eps, create_session, session_info  # type: ignore
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
            self._fallback = GFPGANRestorer(
                device=self._device, bg_upsampler=self._bg, compile_mode=cfg.get("compile", "none")
            )
            self._fallback.prepare(cfg)

    def restore(self, image, cfg: Dict[str, Any]) -> RestoreResult:
        # If ORT not ready, fallback to Torch
        if self._ort is None:
            if self._fallback is None:
                self.prepare(cfg)
            res = self._fallback.restore(image, cfg)
            res.metrics.update(
                {
                    "backend": "torch-fallback",
                    "ort_available_eps": ",".join(self._ort_info.get("available_eps", []) or []),
                    "ort_provider": self._ort_info.get("providers"),
                    "ort_init_sec": self._ort_info.get("init_sec"),
                }
            )
            return res

        # Best-effort ORT inference per cropped face; fallback on any error
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper  # type: ignore

            upscale = int(cfg.get("upscale", 2))
            helper = FaceRestoreHelper(
                upscale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model=str(cfg.get("detector", "retinaface_resnet50")),
                save_ext="png",
                use_parse=bool(cfg.get("use_parse", True)),
                device="cpu",
                model_rootpath="gfpgan/weights",
            )
            helper.clean_all()
            helper.read_image(image)
            helper.get_face_landmarks_5(
                only_center_face=bool(cfg.get("only_center_face", False)),
                eye_dist_threshold=int(cfg.get("eye_dist_threshold", 5)),
            )
            helper.align_warp_face()

            sess = self._ort
            assert sess is not None
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name

            for cropped_face in helper.cropped_faces:
                rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
                x = (rgb.astype("float32") / 255.0) * 2.0 - 1.0
                x = np.transpose(x, (2, 0, 1))[None, ...]
                y = sess.run([out_name], {in_name: x})[0]
                if y.ndim == 4:
                    y = y[0]
                y = np.transpose(y, (1, 2, 0))
                y = np.clip((y + 1.0) / 2.0, 0.0, 1.0)
                bgr = cv2.cvtColor((y * 255.0).astype("uint8"), cv2.COLOR_RGB2BGR)
                helper.add_restored_face(bgr)

            bg_img = None
            if self._bg is not None:
                try:
                    bg_img = self._bg.enhance(image, outscale=upscale)[0]
                except Exception:
                    bg_img = None
            helper.get_inverse_affine(None)
            restored_img = helper.paste_faces_to_input_image(upsample_img=bg_img)
            return RestoreResult(
                input_path=cfg.get("input_path"),
                restored_path=None,
                restored_image=restored_img,
                cropped_faces=[],
                restored_faces=[],
                metrics={
                    "backend": "onnxruntime",
                    "ort_provider": self._ort_info.get("providers"),
                    "ort_init_sec": self._ort_info.get("init_sec"),
                },
            )
        except Exception:
            if self._fallback is None:
                from .gfpgan import GFPGANRestorer  # local import

                self._fallback = GFPGANRestorer(
                    device=self._device, bg_upsampler=self._bg, compile_mode=cfg.get("compile", "none")
                )
                self._fallback.prepare(cfg)
            res = self._fallback.restore(image, cfg)
            res.metrics.update(
                {
                    "backend": "onnxruntime+torch-fallback",
                    "ort_provider": self._ort_info.get("providers"),
                    "ort_init_sec": self._ort_info.get("init_sec"),
                }
            )
            return res
