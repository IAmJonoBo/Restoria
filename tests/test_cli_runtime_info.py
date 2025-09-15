import json
import os
import sys
import tempfile
import subprocess


def _make_input(tmpdir: str):
    os.makedirs(tmpdir, exist_ok=True)
    img = os.path.join(tmpdir, "a.png")
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        arr = (np.random.default_rng(1).random((8, 8, 3)) * 255).astype("uint8")
        Image.fromarray(arr).save(img)
    except Exception:
        with open(img, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
    return img


def _run_cli(extra: list[str]):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    out_dir = tempfile.mkdtemp()
    in_dir = os.path.join(out_dir, "in")
    os.makedirs(in_dir, exist_ok=True)
    _make_input(in_dir)
    cmd = [
        sys.executable,
        "-m",
        "gfpp.cli",
        "run",
        "--input",
        in_dir,
        "--output",
        out_dir,
        "--dry-run",
        "--metrics",
        "fast",
        *extra,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, env=env)
    with open(os.path.join(out_dir, "manifest.json")) as f:
        return json.load(f)


def test_manifest_runtime_section_contains_compile_mode():
    man = _run_cli(["--compile", "none"])  # no actual compile in dry-run
    runtime = (man.get("env") or {}).get("runtime") or {}
    assert "compile_mode" in runtime
    assert "auto_backend" in runtime
    assert "ort_providers" in runtime  # may be None


def test_manifest_runtime_section_auto_backend_flag():
    man = _run_cli(["--auto-backend"])  # should set auto_backend True
    runtime = (man.get("env") or {}).get("runtime") or {}
    assert runtime.get("auto_backend") is True
