from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def autocast_ctx(device_type: str = "cuda", enabled: Optional[bool] = None):
    """Context manager for autocast; no-op if torch missing or CPU without bf16.

    device_type: "cuda" | "cpu" | "mps"
    enabled: None -> autodetect; otherwise force on/off
    """
    try:
        import torch

        if enabled is None:
            enabled = (device_type == "cuda" and torch.cuda.is_available()) or (
                device_type == "cpu" and hasattr(torch.amp, "autocast")
            )
        with torch.autocast(device_type) if enabled else _nullcontext():  # type: ignore[arg-type]
            yield
    except Exception:
        # Fallback no-op
        yield


@contextmanager
def _nullcontext() -> Iterator[None]:
    yield


def to_channels_last(module):
    """Switch module to channels_last memory format when supported.

    Returns the module for chaining; no-op on failure.
    """
    try:
        import torch

        return module.to(memory_format=torch.channels_last)
    except Exception:
        return module


def tile_image(img, tile_size: int = 0):
    """Placeholder for tiling logic. Returns the original image when tile_size<=0.

    Heavy tiling strategies are TODO; keep a small, testable default.
    """
    return img
