import os
import sys
import subprocess
import json

def test_gfpup_doctor_json():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(repo_root, "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "gfpp.cli", "doctor", "--json"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    out = proc.stdout.strip()
    data = json.loads(out)
    assert isinstance(data, dict)
    assert data.get("schema_version") in ("1", 1)
    # basic keys exist (values may be None on minimal env)
    assert "python" in data
    assert "torch" in data
    assert "cuda_available" in data
    assert "onnxruntime_providers" in data
    assert "backends" in data
    assert "suggested_flags" in data
