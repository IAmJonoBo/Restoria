import os
import time

import pytest


def test_gfpp_api_job_smoke(tmp_path):
    try:
        from fastapi.testclient import TestClient
        from services.api.main import app
    except Exception:
        pytest.skip("fastapi not installed")

    client = TestClient(app)

    # Health
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

    # Submit a dry-run job over sample images
    out_dir = tmp_path / "out"
    spec = {
        "input": "inputs/whole_imgs",
        "backend": "gfpgan",
        "background": "none",
        "preset": "natural",
        "compile": "none",
        "metrics": "off",
        "output": str(out_dir),
        "dry_run": True,
    }
    jr = client.post("/jobs", json=spec)
    assert jr.status_code == 200
    job = jr.json()
    jid = job["id"]

    # Poll until done
    t0 = time.time()
    while time.time() - t0 < 10:
        st = client.get(f"/jobs/{jid}")
        assert st.status_code == 200
        js = st.json()
        if js.get("status") == "done":
            break
        time.sleep(0.2)
    assert js.get("status") == "done"

    # Download ZIP of results
    zr = client.get(f"/results/{jid}")
    assert zr.status_code == 200

