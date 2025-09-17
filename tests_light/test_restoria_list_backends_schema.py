import json
import os
import sys
import subprocess


def test_restoria_list_backends_schema():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src") + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "restoria.cli.main", "list-backends", "--json"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    data = json.loads(proc.stdout.strip())
    assert isinstance(data, dict)
    assert data.get("schema_version") == "1"
    b = data.get("backends")
    assert isinstance(b, dict)
    # Ensure at least default backend is present
    assert "gfpgan" in b
