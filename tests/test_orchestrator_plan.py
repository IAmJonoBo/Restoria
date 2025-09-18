from __future__ import annotations


def _fake_quality(niqe=None, brisque=None):
    return {"niqe": niqe, "brisque": brisque}


def test_plan_few_artifacts_routes_gfpgan(monkeypatch, tmp_path):
    # Patch probe_quality to return low metrics
    import gfpp.core.orchestrator as orch

    monkeypatch.setattr("gfpp.core.orchestrator.probe_quality", lambda p: _fake_quality(niqe=6.0, brisque=30))
    # Faces module optional: patch a minimal face_count
    monkeypatch.setattr("gfpp.probe.faces.detect_faces", lambda p: {"face_count": 1}, raising=False)
    plan = orch.plan(str(tmp_path / "img.png"), {"backend": "gfpgan", "weight": 0.5})
    assert plan.backend == "gfpgan"
    assert plan.reason == "few_artifacts"


def test_plan_heavy_degradation_routes_codeformer(monkeypatch, tmp_path):
    import gfpp.core.orchestrator as orch

    monkeypatch.setattr("gfpp.core.orchestrator.probe_quality", lambda p: _fake_quality(niqe=13.0, brisque=60))
    monkeypatch.setattr("gfpp.probe.faces.detect_faces", lambda p: {"face_count": 1}, raising=False)
    plan = orch.plan(str(tmp_path / "img.png"), {"backend": "gfpgan", "weight": 0.5})
    assert plan.backend == "codeformer"
    assert plan.reason == "heavy_degradation"
    # when heavy, weight should be >= 0.6
    assert plan.params.get("weight", 0.0) >= 0.6


def test_plan_heavy_many_faces_biases_gfpgan(monkeypatch, tmp_path):
    import gfpp.core.orchestrator as orch

    monkeypatch.setattr("gfpp.core.orchestrator.probe_quality", lambda p: _fake_quality(niqe=13.0, brisque=60))
    monkeypatch.setattr("gfpp.probe.faces.detect_faces", lambda p: {"face_count": 4}, raising=False)
    plan = orch.plan(str(tmp_path / "img.png"), {"backend": "gfpgan", "weight": 0.5})
    assert plan.backend == "gfpgan"
    assert plan.reason == "heavy_degradation_many_faces"


def test_plan_moderate_defaults_to_gfpgan(monkeypatch, tmp_path):
    import gfpp.core.orchestrator as orch

    # Between thresholds
    monkeypatch.setattr("gfpp.core.orchestrator.probe_quality", lambda p: _fake_quality(niqe=9.0, brisque=45))
    monkeypatch.setattr("gfpp.probe.faces.detect_faces", lambda p: {"face_count": 2}, raising=False)
    plan = orch.plan(str(tmp_path / "img.png"), {"backend": "gfpgan", "weight": 0.4})
    assert plan.backend == "gfpgan"
    assert plan.reason == "moderate_degradation"
    assert abs(plan.params.get("weight") - 0.6) < 1e-9  # deterministic


def test_plan_no_faces_prefers_restoreformer(monkeypatch, tmp_path):
    import gfpp.core.planner as planner

    monkeypatch.setattr("gfpp.core.planner.probe_quality", lambda p: _fake_quality(niqe=10.0, brisque=50))
    monkeypatch.setattr("gfpp.probe.faces.detect_faces", lambda p: {"face_count": 0}, raising=False)
    monkeypatch.setattr(
        "gfpp.core.registry.list_backends",
        lambda include_experimental=False: {"restoreformerpp": {"available": True, "metadata": {}}},
    )
    plan = planner.compute_plan(str(tmp_path / "img.png"), {"backend": "gfpgan", "auto": True})
    assert plan.backend == "restoreformerpp"
    assert plan.reason == "no_faces_detected"
