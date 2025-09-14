from __future__ import annotations

from typing import Any, Dict

from .base import RestoreResult, Restorer


class DiffBIRRestorer(Restorer):
    """EXPERIMENTAL: DiffBIR adapter placeholder.

    Heavy diffusion dependencies; implement behind an extra and feature flag.
    """

    def __init__(self, device: str = "auto", bg_upsampler=None) -> None:
        self._device = device
        self._bg = bg_upsampler

    def prepare(self, cfg: Dict[str, Any]) -> None:
        raise NotImplementedError("DiffBIRRestorer not yet implemented")

    def restore(self, image, cfg: Dict[str, Any]) -> RestoreResult:
        raise NotImplementedError("DiffBIRRestorer not yet implemented")

