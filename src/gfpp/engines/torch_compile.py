from __future__ import annotations

from typing import Optional


def compile_module(module, mode: str = "none"):
    """Wrap a torch.nn.Module with torch.compile when available.

    mode: none|default|max
    Fallbacks to returning the original module on failure.
    """
    if mode in {None, "", "none"}:
        return module
    try:
        import torch

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return module
        map_mode = "default" if mode == "default" else ("max-autotune" if mode == "max" else "default")
        return compile_fn(module, mode=map_mode)
    except Exception:
        return module

