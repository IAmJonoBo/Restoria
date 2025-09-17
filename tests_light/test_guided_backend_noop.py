import os
import sys


def test_guided_backend_no_reference():
    # Ensure src on path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.insert(0, os.path.join(repo_root, "src"))

    from gfpp.restorers.guided import GuidedRestorer  # type: ignore

    # Minimal 1x1 BGR image
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if np is None:
        return  # environment missing numpy; other tests will cover

    img = (np.zeros((1, 1, 3), dtype="uint8"))
    rest = GuidedRestorer(device="cpu", bg_upsampler=None)
    cfg = {"weight": 0.5, "no_download": True}

    res = rest.restore(img, cfg)
    assert res is not None
    assert res.restored_image is not None
    # Metrics should contain guided section with None defaults when reference/ArcFace missing
    assert isinstance(res.metrics, dict)
    guided = res.metrics.get("guided") or {}
    assert "reference" in guided
    assert "arcface_cosine" in guided
    assert "adjusted_weight" in guided
