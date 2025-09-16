from __future__ import annotations

from typing import Any, Dict, Optional

from .base import RestoreResult, Restorer


class GuidedRestorer(Restorer):
    """Reference-guided restoration (stub).

    - If a reference image is provided and dependencies are available, a future
      implementation can align and guide restoration.
    - For now, this returns the input unchanged and records the reference path.
    """

    def __init__(self, device: str = "auto", bg_upsampler=None) -> None:
        self._device = device
        self._bg = bg_upsampler
        self._ref_path: Optional[str] = None

    def prepare(self, cfg: Dict[str, Any]) -> None:
        ref = cfg.get("reference")
        if isinstance(ref, str) and len(ref) > 0:
            self._ref_path = ref

    def restore(self, image: Any, cfg: Dict[str, Any]) -> RestoreResult:
        if self._ref_path is None:
            self.prepare(cfg)
        return RestoreResult(
            input_path=cfg.get("input_path"),
            restored_path=None,
            restored_image=image,
            cropped_faces=[],
            restored_faces=[],
            metrics={"guided": {"reference": self._ref_path}},
        )
