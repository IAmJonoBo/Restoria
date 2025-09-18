from pathlib import Path


def test_restoria_dry_run_ort_backend(tmp_path: Path):
    from restoria.cli.run import run_cmd

    src = tmp_path / "in.png"
    src.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()

    rc = run_cmd(
        [
            "--input",
            str(src),
            "--output",
            str(out),
            "--backend",
            "gfpgan-ort",
            "--metrics",
            "off",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert (out / "manifest.json").exists()
    assert (out / "metrics.json").exists()
