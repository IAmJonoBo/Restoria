import os
import sys
import tempfile


def _write_dummy(path: str):
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        rng = np.random.default_rng(42)
        img = (rng.random((32, 32, 3)) * 255).astype("uint8")
        Image.fromarray(img).save(path)
    except Exception:
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")


def test_enriched_plan_fields():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.insert(0, os.path.join(repo_root, "src"))
    from gfpp.core.orchestrator import plan  # type: ignore

    td = tempfile.mkdtemp()
    img = os.path.join(td, "x.png")
    _write_dummy(img)
    p = plan(img, {"backend": "gfpgan", "weight": 0.5})
    # Basic required attributes
    assert hasattr(p, "quality"), "Plan missing quality field"
    assert hasattr(p, "detail"), "Plan missing detail field"
    assert isinstance(p.quality, dict)
    # Keys may be absent if probe unavailable; accept empty dict
    if p.quality:
        assert set(p.quality.keys()) <= {"niqe", "brisque"}
    # Faces dict optional
    assert hasattr(p, "faces")
    # Detail should include routing rules
    if p.detail and p.detail.get("routing_rules"):
        rr = p.detail.get("routing_rules", {})
        assert {"few_artifacts", "heavy_degradation", "moderate_degradation"}.issubset(rr.keys())
