import os
import subprocess
import sys


def test_gfpp_cli_dry_run_ort(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cmd = [
        sys.executable,
        "-m",
        "src.gfpp.cli",
        "run",
        "--input",
        "inputs/whole_imgs",
        "--backend",
        "gfpgan-ort",
        "--metrics",
        "off",
        "--output",
        str(out),
        "--dry-run",
    ]
    subprocess.run(cmd, check=True)
    assert (out / "manifest.json").is_file()

