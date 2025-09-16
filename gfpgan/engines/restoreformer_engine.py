from typing import Any

from gfpgan.utils import GFPGANer


class RestoreFormerEngine:
    """Engine wrapper for RestoreFormer via GFPGANer arch."""

    def __init__(self, model_path: str, device: Any, **kwargs: Any) -> None:
        # Force arch to RestoreFormer while allowing other GFPGANer kwargs
        kwargs = dict(kwargs)
        kwargs.update({"arch": "RestoreFormer"})
        self.restorer = GFPGANer(model_path=model_path, device=device, **kwargs)
        self.gfpgan = getattr(self.restorer, "gfpgan", None)

    def enhance(self, *args: Any, **kwargs: Any):
        return self.restorer.enhance(*args, **kwargs)
