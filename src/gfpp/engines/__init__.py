"""Execution backends (Torch eager/compile, ONNX Runtime).

All modules must be import-light and only pull heavy deps when actually used.
"""

from .torch_compile import compile_module  # noqa: F401
from .torch_eager import autocast_ctx, to_channels_last  # noqa: F401

__all__ = [
    "compile_module",
    "autocast_ctx",
    "to_channels_last",
]
