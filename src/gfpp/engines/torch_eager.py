from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def autocast_ctx(
    device_type: str = "cuda",
    enabled: Optional[bool] = None,
    *,
    dtype: Optional["object"] = None,
):
    """Context manager for autocast with optional dtype.

    - device_type: "cuda" | "cpu" | "mps"
    - enabled: None -> autodetect; otherwise force on/off
    - dtype: torch.dtype when applicable (e.g., torch.float16 for CUDA, torch.bfloat16 for CPU)
    """
    try:
        import torch

        if enabled is None:
            enabled = (device_type == "cuda" and torch.cuda.is_available()) or (
                device_type == "cpu" and hasattr(torch.amp, "autocast")
            )
        if enabled:
            # Some torch versions require dtype; provide when supplied
            if dtype is not None:
                with torch.autocast(device_type, dtype=dtype):  # type: ignore[arg-type]
                    yield
                    return
            else:
                with torch.autocast(device_type):  # type: ignore[arg-type]
                    yield
                    return
        # Disabled path
        with _nullcontext():
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


def tile_image(img, tile_size: int = 0, tile_overlap: int = 0):
    """Placeholder for tiling logic.

    If tile_size <= 0, returns the original image. In the future, this will
    split and stitch tiles. The parameter is intentionally used to document
    behavior and allow a distinct code path when > 0.
    """
    if tile_size and tile_size > 0:
        try:
            import numpy as np  # type: ignore

            if not isinstance(img, np.ndarray):
                return img
            h, w = img.shape[:2]
            step = max(1, tile_size - max(0, tile_overlap))
            # Prepare accumulation buffers to average overlaps
            out = np.zeros_like(img, dtype=np.float32)
            cnt = np.zeros((h, w, 1), dtype=np.float32)
            for y in range(0, h, step):
                for x in range(0, w, step):
                    y1 = y
                    x1 = x
                    y2 = min(h, y + tile_size)
                    x2 = min(w, x + tile_size)
                    tile = img[y1:y2, x1:x2]
                    # No processing currently; placeholder for per-tile ops
                    out[y1:y2, x1:x2] += tile.astype(np.float32)
                    cnt[y1:y2, x1:x2, :] += 1.0
            cnt[cnt == 0] = 1.0
            merged = (out / cnt).astype(img.dtype)
            return merged
        except Exception:
            return img
    return img
