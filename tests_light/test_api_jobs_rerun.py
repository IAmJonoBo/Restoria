import pytest


def test_jobs_rerun_dry_override(tmp_path):
    try:
        from fastapi.testclient import TestClient
        from services.api.main import app
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

