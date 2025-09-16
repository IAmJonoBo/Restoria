from typing import Any


class CodeFormerEngine:
    def __init__(self, model_path: str, device: Any, **kwargs: Any) -> None:
        # Lazy import heavy backend to keep module import lightweight
        from gfpgan.backends.codeformer_backend import CodeFormerRestorer  # type: ignore

        self.restorer = CodeFormerRestorer(model_path=model_path, device=device, **kwargs)
        self.gfpgan = None  # only GFPGAN has gfpgan attribute

    def enhance(self, *args: Any, **kwargs: Any):  # passthrough
        return self.restorer.enhance(*args, **kwargs)
