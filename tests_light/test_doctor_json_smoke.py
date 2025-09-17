import os
import sys
import subprocess
import json


def test_doctor_json_smoke():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src") + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "gfpp.cli", "doctor", "--json"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    data = json.loads(proc.stdout.strip())
    assert isinstance(data, dict)
    assert data.get("schema_version") == "1"
    backends = data.get("backends") or {}
    assert isinstance(backends, dict)
    # Must at least include stable baseline
    assert "gfpgan" in backends
    # Optional but expected keys we expose by default
    assert "ensemble" in backends
    assert "guided" in backends
