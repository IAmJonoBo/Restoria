import pytest


def test_jobs_list_endpoint(tmp_path):
    try:
        from fastapi.testclient import TestClient
        from services.api.main import app
    except Exception:
        pytest.skip("fastapi not installed")

    client = TestClient(app)
    # Submit one job (dry-run) and list
    spec = {
        "input": "inputs/whole_imgs",
        "backend": "gfpgan",
        "background": "none",
        "metrics": "off",
        "output": str(tmp_path / "out"),
        "dry_run": True,
    }
    jr = client.post("/jobs", json=spec)
    assert jr.status_code == 200
    lst = client.get("/jobs")
    assert lst.status_code == 200
    arr = lst.json()
    assert isinstance(arr, list) and len(arr) >= 1
    assert any(item.get("id") == jr.json()["id"] for item in arr)

