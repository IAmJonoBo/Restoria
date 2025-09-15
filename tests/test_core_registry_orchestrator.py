from __future__ import annotations


def test_registry_lists_backends():
    from gfpp.core.registry import list_backends

    m = list_backends(include_experimental=True)
    assert isinstance(m, dict)
    # Known keys present (availability may vary)
    for k in ["gfpgan", "codeformer", "restoreformerpp", "diffbir", "hypir", "gfpgan-ort"]:
        assert k in m


def test_orchestrator_routes_with_probe_low_quality(monkeypatch):
    from gfpp.core import orchestrator

    # Mock probe returns high NIQE -> heavy degradation
    def _probe_quality(_path: str):
        return {"niqe": 15.0, "brisque": 60.0}

    monkeypatch.setattr(orchestrator, "probe_quality", _probe_quality, raising=True)
    pl = orchestrator.plan("DUMMY_PATH", {"backend": "gfpgan", "weight": 0.5, "background": "realesrgan"})
    assert pl.backend == "codeformer"  # heavy degradation
    assert 0.6 <= float(pl.params.get("weight", 0.0)) <= 1.0


def test_orchestrator_routes_with_probe_good_quality(monkeypatch):
    from gfpp.core import orchestrator

    def _probe_quality(_path: str):
        return {"niqe": 6.5, "brisque": 30.0}

    monkeypatch.setattr(orchestrator, "probe_quality", _probe_quality, raising=True)
    pl = orchestrator.plan("DUMMY_PATH", {"backend": "gfpgan", "weight": 0.4, "background": "realesrgan"})
    assert pl.backend == "gfpgan"
    assert 0.0 <= float(pl.params.get("weight", 0.0)) <= 1.0


def test_orchestrator_routes_with_probe_moderate(monkeypatch):
    from gfpp.core import orchestrator

    def _probe_quality(_path: str):
        return {"niqe": 9.0, "brisque": 45.0}

    monkeypatch.setattr(orchestrator, "probe_quality", _probe_quality, raising=True)
    pl = orchestrator.plan("DUMMY_PATH", {"backend": "gfpgan", "weight": 0.5, "background": "realesrgan"})
    assert pl.backend == "gfpgan"
    assert abs(float(pl.params.get("weight", 0.0)) - 0.6) < 1e-6


def test_orchestrator_no_probe_fallback(monkeypatch):
    from gfpp.core import orchestrator

    def _probe_quality(_path: str):  # simulate unavailability
        return None

    monkeypatch.setattr(orchestrator, "probe_quality", _probe_quality, raising=True)
    pl = orchestrator.plan("DUMMY_PATH", {"backend": "codeformer", "weight": 0.8, "background": "realesrgan"})
    # Should keep the provided backend and weight when no quality signal
    assert pl.backend == "codeformer"
    assert abs(float(pl.params.get("weight", 0.0)) - 0.8) < 1e-6
