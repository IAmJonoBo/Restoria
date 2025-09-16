import os
import pytest

if not os.environ.get("RUN_API_TESTS"):
    pytest.skip("API tests are skipped by default. Set RUN_API_TESTS=1 to enable.", allow_module_level=True)


def test_jobs_rerun_dry_override(tmp_path):
    try:
        from services.api.main import app
        from fastapi.testclient import TestClient
    except Exception:
        pytest.skip("fastapi not installed")

    client = TestClient(app)
    out_dir = tmp_path / "out"
    spec = {
        "input": "inputs/whole_imgs",
        "backend": "gfpgan",
        "background": "none",
        "metrics": "off",
        "output": str(out_dir),
        "dry_run": True,
    }
    jr = client.post("/jobs", json=spec)
    assert jr.status_code == 200
    jid = jr.json()["id"]

    # Rerun with metrics=full (still dry_run)
    rr = client.post(f"/jobs/{jid}/rerun", json={"metrics": "full", "dry_run": True})
    assert rr.status_code == 200
    j2 = rr.json()
    assert j2["id"] != jid
