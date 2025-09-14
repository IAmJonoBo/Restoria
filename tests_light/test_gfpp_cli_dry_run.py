import os
import subprocess
import sys


def test_gfpp_cli_dry_run(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # Run the CLI in dry-run mode using the src package path
    cmd = [sys.executable, "-m", "src.gfpp.cli", "run", "--input", "inputs/whole_imgs", "--backend", "gfpgan", "--metrics", "off", "--output", str(out), "--dry-run"]
    subprocess.run(cmd, check=True)
    # Expect manifest.json present
    assert (out / "manifest.json").is_file()

