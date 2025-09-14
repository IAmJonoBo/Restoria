from __future__ import annotations

from typing import Any, Dict

from .base import RestoreResult, Restorer


class HYPIRRestorer(Restorer):
    """EXPERIMENTAL: HYPIR adapter placeholder.

    Heavy dependencies; implement behind an extra and feature flag.
    """

    def __init__(self, device: str = "auto", bg_upsampler=None) -> None:
        self._device = device
        self._bg = bg_upsampler

    def prepare(self, cfg: Dict[str, Any]) -> None:
        raise NotImplementedError("HYPIRRestorer not yet implemented")

    def restore(self, image, cfg: Dict[str, Any]) -> RestoreResult:
        raise NotImplementedError("HYPIRRestorer not yet implemented")
