from __future__ import annotations

from typing import Any, Protocol, Tuple


class RestorerEngine(Protocol):
    """Protocol describing a minimal face restorer engine.

    Current engines expose an `enhance` method compatible with GFPGANer.
    This protocol exists to improve type clarity and future-proof evolution
    toward prepare() -> run() -> compose() style APIs without forcing refactors.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - interface only
        ...

    def enhance(
        self,
        img,  # numpy ndarray (BGR)
        has_aligned: bool = False,
        only_center_face: bool = False,
        paste_back: bool = True,
        weight: float = 0.5,
        eye_dist_threshold: int = 5,
    ) -> Tuple[list, list, Any]:  # (cropped_faces, restored_faces, restored_img)
        ...


__all__ = ["RestorerEngine"]

