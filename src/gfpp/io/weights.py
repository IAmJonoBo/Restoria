from __future__ import annotations

from typing import Tuple


def ensure_weight(model_name: str, *, no_download: bool = False) -> Tuple[str, str | None]:
    """Resolve model weight path and sha256 via central logic.

    Delegates to gfpgan.weights.resolve_model_weight to avoid duplication.
    """
    from gfpgan.weights import resolve_model_weight

    return resolve_model_weight(model_name, no_download=no_download)
