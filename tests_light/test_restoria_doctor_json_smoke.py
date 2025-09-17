import json
import os
import sys
import subprocess


def test_restoria_doctor_json_smoke():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src") + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "restoria.cli.main", "doctor", "--json"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    data = json.loads(proc.stdout.strip())
    assert isinstance(data, dict)
    # Expected keys; values may be None depending on environment
    assert "python" in data
    assert "cuda_available" in data
    assert "onnxruntime_providers" in data
    assert "suggested_flags" in data
