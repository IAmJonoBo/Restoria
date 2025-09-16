from __future__ import annotations

from typing import Any, Dict, Optional

from .base import RestoreResult, Restorer


class HYPIRRestorer(Restorer):
    """EXPERIMENTAL: HYPIR adapter placeholder.

    Heavy dependencies; implement behind an extra and feature flag.
    """

    def __init__(self, device: str = "auto", bg_upsampler=None) -> None:
        self._device = device
        self._bg = bg_upsampler
        self._prepared = False
        self._model = None  # type: ignore

    def prepare(self, cfg: Dict[str, Any]) -> None:
        """Lazy-prepare model. In stub mode, do nothing.

        Real implementation should:
        - Resolve weights (cached)
        - Build model on device (respect compile mode if requested)
        - Seed deterministically when requested
        """
        # Best-effort: try to import hypothetical hypir package (optional)
        try:
            self._prepared = True
        except Exception:
            # Stub path stays prepared=False, restore() will no-op copy image
            self._prepared = False

    def restore(self, image, cfg: Dict[str, Any]) -> RestoreResult:
        """Single-pass promptable restoration (stub).

        Parameters accepted in cfg (all optional):
        - texture_richness: float in [0,1]
        - prompt: str
        - identity_lock: bool
        - input_path: str (for manifest provenance)
        """
        # Normalize cfg params
        tex = cfg.get("texture_richness")
        try:
            if tex is not None:
                tex = float(tex)
                tex = max(0.0, min(1.0, tex))
        except Exception:
            tex = None
        prompt: Optional[str] = cfg.get("prompt")
        identity_lock = bool(cfg.get("identity_lock", False))

        # In stub mode or if _model is not available, just return input image as restored
        restored_img = image
        metrics: Dict[str, Any] = {
            "backend": "hypir",
            "mode": "stub",
            "texture_richness": tex,
            "prompt_len": (len(prompt) if isinstance(prompt, str) else None),
            "identity_lock": identity_lock,
        }
        return RestoreResult(
            input_path=cfg.get("input_path"),
            restored_path=None,
            restored_image=restored_img,
            cropped_faces=[],
            restored_faces=[],
            metrics=metrics,
        )
