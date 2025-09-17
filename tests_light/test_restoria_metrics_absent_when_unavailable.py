import json
import os
import tempfile


def test_restoria_metrics_absent_when_unavailable(monkeypatch):
    # Ensure metrics helpers return None to mimic unavailable deps
    from restoria.cli import run as runmod

    monkeypatch.setattr(runmod, "_maybe_arcface", lambda args: None)
    monkeypatch.setattr(runmod, "_maybe_lpips_dists", lambda args: (None, None))

    # Also bypass image loading to avoid heavy deps in light suite
    monkeypatch.setattr(runmod, "_load_image", lambda p: object())

    rc = 1
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.png")
        open(src, "wb").close()
        out = os.path.join(td, "out")
        rc = runmod.run_cmd([
            "--input", src,
            "--output", out,
            "--backend", "gfpgan",
            "--metrics", "full",
        ])
        assert rc == 0
        man = json.load(open(os.path.join(out, "manifest.json")))
        assert isinstance(man.get("results"), list)
        # For each result, metrics dict should either be empty or not contain our optional keys
        for rec in man.get("results"):
            m = rec.get("metrics") or {}
            assert "arcface_cosine" not in m
            assert "lpips_alex" not in m
            assert "dists" not in m
