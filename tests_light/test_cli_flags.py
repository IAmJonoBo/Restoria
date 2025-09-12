import subprocess
import sys


def run_ok(args):
    subprocess.run([sys.executable, "inference_gfpgan.py", "--dry-run", *args], check=True)


def test_dry_run_device_no_download():
    run_ok(["--device", "cpu", "--no-download", "-v", "1.4"])
