import subprocess
import sys


def run_ok(args):
    subprocess.run([sys.executable, "inference_gfpgan.py", "--dry-run", *args], check=True)


def test_backends_parse_all():
    for backend in ["gfpgan", "restoreformer", "restoreformerpp", "codeformer"]:
        run_ok(["--backend", backend, "-v", "1.4"])  # version value ignored for non-gfpgan


def test_compile_flag_parses():
    run_ok(["--compile", "default", "-v", "1.4"])
    run_ok(["--compile", "max", "-v", "1.4"])


def test_metrics_flag_parses():
    for m in ["none", "id", "lpips", "both"]:
        run_ok(["--metrics", m, "-v", "1.4"])  # no effect on dry-run
