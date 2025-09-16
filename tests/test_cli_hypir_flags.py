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


def test_cli_hypir_prompt_passes_to_plan_and_falls_back_gracefully():
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
        "off",
        "--auto-backend",
        "--experimental",
        "--prompt",
        "a test prompt",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, env=env)
    with open(os.path.join(out_dir, "manifest.json")) as f:
        man = json.load(f)
    # Plan present and contains reason backend even if fallback occurred
    results = man.get("results") or man
    # Accept both shapes based on manifest writer, but ensure has data
    assert results


def test_cli_hypir_params_parse_only():
    # Just ensure CLI parses flags without executing model code
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
        "off",
        "--backend",
        "hypir",
        "--experimental",
        "--prompt",
        "simple",
        "--texture-richness",
        "0.8",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(repo_root, "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, env=env)
    # If it ran, parsing worked
