import os
import sys
import subprocess
import json


def test_gfpup_list_backends_json(tmp_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(repo_root, "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "gfpp.cli", "list-backends", "--json"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    out = proc.stdout.strip()
    data = json.loads(out)
    assert isinstance(data, dict)
    assert "experimental" in data
    assert "backends" in data
    assert isinstance(data["backends"], dict)
