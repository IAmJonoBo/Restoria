def test_orchestrator_sets_detector_many_faces(monkeypatch):
    from gfpp.core import orchestrator as orch

    def fake_detect_faces(_path):
        return {"count": 4, "sizes": [40, 42, 38, 45]}

    monkeypatch.setattr("gfpp.probe.faces.detect_faces", fake_detect_faces, raising=False)
    plan = orch.plan("/dev/null", {"backend": "gfpgan"})
    assert plan.params.get("detector") == "scrfd"


def test_orchestrator_sets_detector_small_faces(monkeypatch):
    from gfpp.core import orchestrator as orch

    def fake_detect_faces(_path):
        return {"count": 1, "sizes": [60, 62, 58]}

    monkeypatch.setattr("gfpp.probe.faces.detect_faces", fake_detect_faces, raising=False)
    plan = orch.plan("/dev/null", {"backend": "gfpgan"})
    assert plan.params.get("detector") == "scrfd"


def test_orchestrator_sets_detector_large_faces(monkeypatch):
    from gfpp.core import orchestrator as orch

    def fake_detect_faces(_path):
        return {"count": 1, "sizes": [120, 130]}

    monkeypatch.setattr("gfpp.probe.faces.detect_faces", fake_detect_faces, raising=False)
    plan = orch.plan("/dev/null", {"backend": "gfpgan"})
    # For larger faces and few count, prefer retinaface
    assert plan.params.get("detector") == "retinaface_resnet50"
