from typing import Any

import torch

from gfpgan.backends.codeformer_backend import CodeFormerRestorer


class CodeFormerEngine:
    def __init__(self, model_path: str, device: torch.device, **kwargs: Any) -> None:
        self.restorer = CodeFormerRestorer(model_path=model_path, device=device, **kwargs)
        self.gfpgan = None  # only GFPGAN has gfpgan attribute

    def enhance(self, *args: Any, **kwargs: Any):  # passthrough
        return self.restorer.enhance(*args, **kwargs)

