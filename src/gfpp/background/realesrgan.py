from __future__ import annotations

from typing import Optional  # noqa: F401


def build_realesrgan(device: str = "cuda", tile: int = 400, precision: str = "auto"):
    """Return a RealESRGANer instance or None if dependency missing.

    precision: auto|fp16|fp32
    """
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
        from realesrgan import RealESRGANer  # type: ignore

        dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        if dev == "cpu":
            return None
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        half = True if precision == "auto" else (precision == "fp16")
        return RealESRGANer(
            scale=2,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            model=model,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=half,
        )
    except Exception:
        return None
