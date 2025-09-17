import os
import sys
import subprocess
import json


def test_list_backends_json_includes_guided_and_ensemble():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(repo_root, "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "gfpp.cli", "list-backends", "--json"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    data = json.loads(proc.stdout.strip())
    assert isinstance(data, dict)
    backends = data.get("backends") or {}
    # Presence in keys is sufficient; availability may be false depending on env
    assert "ensemble" in backends
    assert "guided" in backends
