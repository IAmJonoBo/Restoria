from __future__ import annotations

from typing import Optional


class ArcFaceIdentity:
    """Batched ArcFace cosine similarity wrapper with graceful fallback.

    Falls back to a dummy implementation when ArcFace weights are unavailable.
    """

    def __init__(self, no_download: bool = False) -> None:
        self.model = None
        try:
            # Reuse existing implementation where possible
            from gfpgan.metrics import try_load_arcface  # type: ignore

            self.model = try_load_arcface(no_download=no_download)
        except Exception:
            self.model = None

    def available(self) -> bool:
        return self.model is not None

    def cosine_from_paths(self, a_path: str, b_path: str) -> Optional[float]:
        if self.model is None:
            return None
        try:
            from gfpgan.metrics import identity_cosine_from_paths  # type: ignore

            return identity_cosine_from_paths(a_path, b_path, self.model)
        except Exception:
            return None
