from __future__ import annotations

import numpy as np

from src.gfpp.restorers.gfpgan import GFPGANRestorer


def test_gfpgan_autocast_tiling_stub_safe():
    # Create a tiny dummy image
    img = np.zeros((4, 4, 3), dtype=np.uint8)

    rest = GFPGANRestorer(device="cpu", bg_upsampler=None, compile_mode="none")
    # Do not call prepare explicitly to allow lazy/stub path
    cfg = {
        "weight": 0.5,
        "precision": "fp16",  # should be a no-op on CPU
        "tile": 32,            # triggers tile_image path (returns a copy)
        "tile_overlap": 8,
        "input_path": None,
    }
    res = rest.restore(img, cfg)

    assert res is not None
    assert res.restored_image is not None
    assert isinstance(res.restored_image, np.ndarray)
