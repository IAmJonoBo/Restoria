import time

import pytest


def test_ws_stream_dry_run(tmp_path):
    try:
        from services.api.main import app
        from fastapi.testclient import TestClient
    except Exception:
        pytest.skip("fastapi not installed")

    client = TestClient(app)

    # Submit a job
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

    # Connect WebSocket and consume events until eof
    with client.websocket_connect(f"/jobs/{jid}/stream") as ws:
        saw_status = False
        saw_image = False
        saw_eof = False
        t0 = time.time()
        while time.time() - t0 < 10:
            msg = ws.receive_json()
            if msg.get("type") == "status":
                saw_status = True
            if msg.get("type") == "image":
                saw_image = True
            if msg.get("type") == "eof":
                saw_eof = True
                break
        assert saw_status and saw_image and saw_eof
