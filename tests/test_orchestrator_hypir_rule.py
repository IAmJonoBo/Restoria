from __future__ import annotations


def _fake_quality(niqe=None, brisque=None):
    return {"niqe": niqe, "brisque": brisque}


def test_hypir_routes_with_prompt_when_experimental(monkeypatch, tmp_path):
    import gfpp.core.orchestrator as orch

    # Moderate degradation
    monkeypatch.setattr("gfpp.core.orchestrator.probe_quality", lambda p: _fake_quality(niqe=9.0, brisque=45))
    # Few faces
    monkeypatch.setattr("gfpp.probe.faces.detect_faces", lambda p: {"face_count": 1}, raising=False)
    # Pretend hypir is available
    monkeypatch.setattr(
        "gfpp.core.orchestrator.list_backends",
        lambda include_experimental=True: {"hypir": True},
        raising=False,
    )
    pl = orch.plan(
        str(tmp_path / "img.png"),
        {"backend": "gfpgan", "weight": 0.5, "experimental": True, "prompt": "restore details"},
    )
    assert pl.backend == "hypir"
    assert pl.reason == "experimental_hypir_prompt"


def test_hypir_routes_on_moderate_when_experimental(monkeypatch, tmp_path):
    import gfpp.core.orchestrator as orch

    # Moderate degradation
    monkeypatch.setattr("gfpp.core.orchestrator.probe_quality", lambda p: _fake_quality(niqe=9.0, brisque=45))
    # Few faces
    monkeypatch.setattr("gfpp.probe.faces.detect_faces", lambda p: {"face_count": 2}, raising=False)
    # Pretend hypir is available
    monkeypatch.setattr(
        "gfpp.core.orchestrator.list_backends",
        lambda include_experimental=True: {"hypir": True},
        raising=False,
    )
    pl = orch.plan(str(tmp_path / "img.png"), {"backend": "gfpgan", "weight": 0.5, "experimental": True})
    assert pl.backend == "hypir"
    assert pl.reason == "experimental_hypir_moderate"
