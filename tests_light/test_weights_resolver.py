import hashlib


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_resolve_prefers_env_dir(tmp_path, monkeypatch):
    # Create a fake weight file in a custom weights dir
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    fname = weights_dir / "GFPGANv1.4.pth"
    data = b"fake-weights"
    fname.write_bytes(data)
    monkeypatch.setenv("GFPGAN_WEIGHTS_DIR", str(weights_dir))

    from gfpgan.weights import resolve_model_weight

    path, digest = resolve_model_weight("GFPGANv1.4", no_download=True)
    assert path == str(fname)
    assert digest == sha256(str(fname))


def test_offline_raises_when_missing(tmp_path, monkeypatch):
    # Empty custom dir + offline -> should raise
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    monkeypatch.setenv("GFPGAN_WEIGHTS_DIR", str(weights_dir))

    from gfpgan.weights import resolve_model_weight

    try:
        resolve_model_weight("GFPGANv1.3", no_download=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError in offline mode when file missing")
