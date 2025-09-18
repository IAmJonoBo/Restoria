import types
import sys


def test_probe_prefers_scrfd_when_available(monkeypatch, tmp_path):
    # Create a dummy image file path; detector functions only check None vs array, so we bypass by mocking
    dummy_path = str(tmp_path / "img.jpg")

    # Mock cv2.imread to return a non-None ndarray-like object
    class _Arr:
        shape = (128, 128, 3)
        dtype = type("_D", (), {})

    def fake_imread(_):
        return _Arr()

    # Mock insightface FaceAnalysis
    class _Face:
        def __init__(self, bbox):
            self.bbox = bbox

    class _App:
        def __init__(self, *args, **kwargs):
            # test stub: accepts arbitrary constructor arguments
            pass
        def prepare(self, *args, **kwargs):
            return None

        def get(self, img):
            # Return two faces with small sizes to exercise sizes aggregation
            return [_Face([10, 10, 50, 50]), _Face([20, 20, 45, 60])]

    # no mod_insightface; we inject modules directly via sys.modules stubs below

    # Stub cv2 and insightface.app via sys.modules
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(imread=fake_imread))
    # Ensure parent package exists
    insight_pkg = types.ModuleType("insightface")
    insight_app = types.ModuleType("insightface.app")
    # Assign attribute directly
    insight_app.FaceAnalysis = _App  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "insightface", insight_pkg)
    monkeypatch.setitem(sys.modules, "insightface.app", insight_app)

    from gfpp.probe.faces import detect_faces

    stats = detect_faces(dummy_path)
    assert isinstance(stats, dict)
    assert stats.get("count") == 2
    assert stats.get("sizes") and all(isinstance(s, int) for s in stats["sizes"])  # type: ignore[index]


def test_probe_falls_back_to_facexlib(monkeypatch, tmp_path):
    dummy_path = str(tmp_path / "img.jpg")

    class _Arr:
        shape = (100, 100, 3)

    def fake_imread(_):
        return _Arr()

    # Mock facexlib retinaface
    class _Det:
        def detect(self, img, threshold=0.6):
            # Provide one bbox using expected (x1, y1, x2, y2, score)
            return [[0, 0, 10, 20, 0.9]], None

    # Stub cv2 and facexlib.detection.retinaface; do not provide insightface.app to force fallback
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(imread=fake_imread))
    # Ensure facexlib packages exist
    facex_pkg = types.ModuleType("facexlib")
    facex_det_pkg = types.ModuleType("facexlib.detection")
    facex_ret_pkg = types.ModuleType("facexlib.detection.retinaface")
    facex_ret_pkg.RetinaFace = _Det  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "facexlib", facex_pkg)
    monkeypatch.setitem(sys.modules, "facexlib.detection", facex_det_pkg)
    monkeypatch.setitem(sys.modules, "facexlib.detection.retinaface", facex_ret_pkg)
    # Ensure insightface.app is absent
    for k in tuple(sys.modules.keys()):
        if k.startswith("insightface"):
            sys.modules.pop(k, None)

    from gfpp.probe.faces import detect_faces

    stats = detect_faces(dummy_path)
    assert isinstance(stats, dict)
    assert stats.get("count") == 1
    assert stats.get("sizes") == [20]
