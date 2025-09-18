from __future__ import annotations

from gfpp.core.planner import Plan


def test_restoria_plan_respects_auto(monkeypatch):
    from restoria.core import planner as rest_planner

    fake = Plan(backend="codeformer", params={"weight": 0.7}, postproc={}, reason="heavy_degradation", confidence=0.8)

    monkeypatch.setattr(rest_planner, "base_compute_plan", lambda _path, _opts: fake)

    forced = rest_planner.compute_plan("/tmp/img.png", {"backend": "gfpgan", "auto": False})
    assert forced.backend == "gfpgan"
    assert forced.reason == "user_selected"

    auto = rest_planner.compute_plan("/tmp/img.png", {"backend": "gfpgan", "auto": True})
    assert auto.backend == "codeformer"
    assert auto.reason == fake.reason


def test_restoria_plan_threads_hints(monkeypatch):
    from restoria.core import planner as rest_planner

    fake = Plan(backend="gfpgan", params={}, postproc={}, reason="few_artifacts", confidence=0.9)
    monkeypatch.setattr(rest_planner, "base_compute_plan", lambda _path, _opts: fake)

    plan = rest_planner.compute_plan(
        "/tmp/img.png",
        {
            "backend": "gfpgan",
            "compile": True,
            "ort_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        },
    )
    assert plan.params["compile"] is True
    assert plan.params["ort_providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
