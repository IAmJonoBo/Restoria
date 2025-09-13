import os
import tempfile

import numpy as np


def _write_png(path: str, seed: int = 0) -> None:
    import cv2

    rng = np.random.default_rng(seed)
    img = (rng.random((16, 16, 3)) * 255).astype("uint8")
    cv2.imwrite(path, img)


def test_metrics_helpers_graceful():
    from gfpgan.metrics import (
        identity_cosine_from_paths,
        lpips_from_paths,
        try_load_arcface,
        try_lpips_model,
    )

    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "a.png")
        b = os.path.join(td, "b.png")
        _write_png(a, seed=1)
        _write_png(b, seed=2)

        # ArcFace model may be unavailable; ensure helper doesn't crash
        id_model = try_load_arcface(no_download=True)
        if id_model is None:
            assert identity_cosine_from_paths(a, b, id_model) is None
        else:
            score = identity_cosine_from_paths(a, a, id_model)
            assert isinstance(score, float)

        # LPIPS model may be unavailable; ensure helper doesn't crash
        lpips_model = try_lpips_model()
        if lpips_model is None:
            assert lpips_from_paths(a, b, lpips_model) is None
        else:
            d = lpips_from_paths(a, a, lpips_model)
            assert isinstance(d, float)

