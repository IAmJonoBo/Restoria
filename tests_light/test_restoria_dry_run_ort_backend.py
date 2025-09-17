import os
import tempfile


def test_restoria_dry_run_ort_backend():
    from restoria.cli.run import run_cmd

    with tempfile.TemporaryDirectory() as td:
        # Prepare a minimal input file; dry-run will copy when decode unavailable
        src = os.path.join(td, "in.png")
        open(src, "wb").close()
        out = os.path.join(td, "out")
        rc = run_cmd([
            "--input", src,
            "--output", out,
            "--backend", "gfpgan-ort",
            "--metrics", "off",
            "--dry-run",
        ])
        assert rc == 0
        assert os.path.exists(os.path.join(out, "manifest.json"))
        assert os.path.exists(os.path.join(out, "metrics.json"))
