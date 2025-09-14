from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class RestoreResult:
    input_path: Optional[str]
    restored_path: Optional[str]
    restored_image: Optional[Any]
    cropped_faces: list[str]
    restored_faces: list[str]
    metrics: Dict[str, Any]


class Restorer(Protocol):
    def prepare(self, cfg: Dict[str, Any]) -> None: ...

    def restore(self, image: Any, cfg: Dict[str, Any]) -> RestoreResult: ...
