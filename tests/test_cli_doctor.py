import os
import sys
import subprocess


def test_gfpup_doctor_runs_and_lists_backends(tmp_path):
    """Smoke-test the 'gfpup doctor' subcommand.

    This should exit 0 and print a Backends section regardless of optional deps.
    Avoids heavy imports and downloads; runs with repo src on PYTHONPATH.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(repo_root, "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    # Run as a module to ensure correct entry
    cmd = [sys.executable, "-m", "gfpp.cli", "doctor"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)

    # Must not crash
    assert proc.returncode == 0, f"doctor failed: {proc.stderr}"

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Minimal signal: it should print a Backends section from the registry
    assert "Backends:" in out
