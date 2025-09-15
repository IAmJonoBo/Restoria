from __future__ import annotations

import os
import tempfile

import numpy as np  # type: ignore

from gfpp.metrics import ArcFaceIdentity, LPIPSMetric, DISTSMetric, NoRefQuality
from gfpp.probe.quality import probe_quality


def _make_temp_image(color=(128, 128, 128)) -> str:
    td = tempfile.mkdtemp()
    path = os.path.join(td, "im.png")
    arr = np.full((32, 32, 3), color, dtype=np.uint8)
    # Try OpenCV first for consistency with metric loaders; otherwise use PIL
    try:  # pragma: no cover - branch depends on optional cv2
        import cv2  # type: ignore

        cv2.imwrite(path, arr)
        return path
    except Exception:  # fallback: PIL
        try:
            from PIL import Image  # type: ignore

            Image.fromarray(arr[:, :, ::-1]).save(path)  # convert BGR->RGB
            return path
        except Exception:
            # As a last resort, write raw bytes (metrics will fail gracefully)
            with open(path, "wb") as f:
                f.write(b"\x00" * 10)
            return path


def test_arcface_wrapper_graceful():
    arc = ArcFaceIdentity(no_download=True)
    # availability may be False locally; just assert no exception and value is None when unavailable
    a = _make_temp_image()
    b = _make_temp_image((129, 129, 129))
    sim = arc.cosine_from_paths(a, b)
    if arc.available():
        assert sim is None or isinstance(sim, float)
    else:
        assert sim is None


def test_lpips_wrapper_graceful():
    lp = LPIPSMetric()
    a = _make_temp_image()
    d = lp.distance_from_paths(a, a)
    if lp.available():
        assert d is None or isinstance(d, float)
    else:
        assert d is None


def test_dists_wrapper_graceful():
    dm = DISTSMetric()
    a = _make_temp_image()
    d = dm.distance_from_paths(a, a)
    if dm.available():
        assert d is None or isinstance(d, float)
    else:
        assert d is None


def test_noref_quality_wrapper():
    nr = NoRefQuality()
    a = _make_temp_image()
    scores = nr.score(a)
    # Scores may be empty if neither metric available
    assert isinstance(scores, dict)
    for k, v in scores.items():
        assert k in {"niqe", "brisque"}
        assert isinstance(v, float)


def test_probe_quality_best_effort(monkeypatch):
    # Force underlying metrics to return None to exercise fallback
    from gfpp.probe import quality as qmod

    def _niqe_fail(_):
        return None

    def _brisque_fail(_):
        return None

    monkeypatch.setattr(qmod, "_niqe", _niqe_fail, raising=True)
    monkeypatch.setattr(qmod, "_brisque", _brisque_fail, raising=True)
    res = probe_quality("DUMMY")
    assert res is None
