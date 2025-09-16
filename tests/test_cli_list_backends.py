import os
import sys
import subprocess


def test_gfpup_list_backends_defaults(tmp_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(repo_root, "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "gfpp.cli", "list-backends"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Should print a header and at least gfpgan in non-experimental set
    assert "Backends (experimental=off):" in out
    assert "gfpgan" in out


def test_gfpup_list_backends_all_verbose(tmp_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(repo_root, "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "gfpp.cli", "list-backends", "--all", "--verbose"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Should include experimental on and list names with availability markers
    assert "Backends (experimental=on):" in out
    assert "available" in out or "missing" in out
