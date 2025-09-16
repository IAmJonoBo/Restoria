import os
import pytest

if not os.environ.get("RUN_API_TESTS"):
    pytest.skip("API tests are skipped by default. Set RUN_API_TESTS=1 to enable.", allow_module_level=True)


@pytest.mark.skipif("fastapi" not in globals(), reason="fastapi not installed")
def test_api_health_and_restore_dry_run():
    try:
        from fastapi.testclient import TestClient
        from gfpgan.api.server import app
    except Exception:  # pragma: no cover - skip when fastapi not installed
        pytest.skip("fastapi not installed")

    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"

    for backend in ("gfpgan", "restoreformer", "codeformer"):
        files = {"files": ("x.png", b"fake", "image/png")}
        resp = client.post(f"/restore?backend={backend}&dry_run=true", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("dry_run") is True
        assert isinstance(data.get("results"), list)
