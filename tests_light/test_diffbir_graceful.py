from __future__ import annotations

import numpy as np

from src.gfpp.restorers.diffbir import DiffBIRRestorer


def test_diffbir_graceful_return_input_when_unavailable():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    rest = DiffBIRRestorer(device="cpu", bg_upsampler=None)
    res = rest.restore(img, cfg={"input_path": None})
    assert res is not None
    assert res.restored_image is not None
