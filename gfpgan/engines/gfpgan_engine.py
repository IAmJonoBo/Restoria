from typing import Any

import torch

from gfpgan.utils import GFPGANer


class GFPGANEngine:
    """Adapter engine around GFPGANer to provide a consistent interface."""

    def __init__(self, model_path: str, device: torch.device, **kwargs: Any) -> None:
        self.restorer = GFPGANer(model_path=model_path, device=device, **kwargs)
        self.gfpgan = getattr(self.restorer, "gfpgan", None)  # for optional torch.compile

    def enhance(self, *args: Any, **kwargs: Any):  # passthrough
        return self.restorer.enhance(*args, **kwargs)

