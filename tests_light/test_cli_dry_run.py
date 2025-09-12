import subprocess
import sys


def test_cli_dry_run():
    # Validate CLI parses and returns 0 without heavy work
    cmd = [sys.executable, "inference_gfpgan.py", "--dry-run", "-v", "1.4"]
    subprocess.run(cmd, check=True)
