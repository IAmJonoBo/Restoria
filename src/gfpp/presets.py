from __future__ import annotations

from typing import Any, Dict


def apply_preset(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a named preset to config (non-destructive).

    Supported: 'document' (stub) → favor denoising and background SR later.
    """
    out = dict(cfg)
    if name == "document":
        # Favor lower weight and keep background upsampler engaged
        out.setdefault("weight", 0.3)
        out.setdefault("detector", "retinaface_resnet50")
        out.setdefault("use_parse", True)
    return out
