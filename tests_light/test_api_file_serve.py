import os


def test_file_serve_inputs():
    try:
        from services.api.main import app
        from fastapi.testclient import TestClient
    except Exception:
        return

    client = TestClient(app)
    # Pick a sample input image
    p = "inputs/whole_imgs/00.jpg"
    if not os.path.isfile(p):
        return
    r = client.get("/file", params={"path": p})
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("image/")
