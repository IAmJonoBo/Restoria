from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

from gfpp.core.planner import Plan
from services.api import jobs
from services.api.schemas import JobSpec

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


@pytest.mark.parametrize("auto_flag", [False, True])
def test_jobmanager_dry_run_records_plan(monkeypatch, tmp_path, auto_flag: bool) -> None:
    def fake_compute_plan(_path: str, opts: dict[str, object]) -> Plan:
        if opts.get("auto"):
            return Plan(backend="codeformer", params={}, postproc={}, reason="auto_route", confidence=0.9)
        return Plan(backend=str(opts.get("backend", "gfpgan")), params={}, postproc={}, reason="user_selected", confidence=0.5)

    monkeypatch.setattr(jobs.rest_planner, "compute_plan", fake_compute_plan)

    inp = tmp_path / "input.png"
    inp.write_bytes(_TINY_PNG)
    out_dir = tmp_path / "out"

    spec = JobSpec(
        input=str(inp),
        backend="gfpgan",
        background="none",
        quality="balanced",
        preset="natural",
        compile="none",
        seed=None,
        deterministic=False,
        metrics="off",
        output=str(out_dir),
        dry_run=True,
        model_path_onnx=None,
        auto_backend=auto_flag,
    )

    manager = jobs.JobManager()
    job = manager.create(spec)
    monkeypatch.setenv("NB_CI_SMOKE", "1")
    asyncio.run(manager.run(job.id))

    metrics_path = Path(job.results_path or out_dir) / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["metrics"], "expected at least one metric entry"
    entry = payload["metrics"][0]
    plan_info = entry.get("plan", {})
    expected_backend = "codeformer" if auto_flag else "gfpgan"
    expected_reason = "auto_route" if auto_flag else "user_selected"
    assert job.results, "job should cache results for sync restore"
    assert job.results[0]["plan"]["backend"] == expected_backend
    assert plan_info.get("backend") == expected_backend
    assert plan_info.get("reason") == expected_reason
    metrics = entry.get("metrics", {})
    assert metrics.get("plan_backend") == expected_backend
    assert metrics.get("plan_reason") == expected_reason
