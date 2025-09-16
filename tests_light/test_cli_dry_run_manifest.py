import os
import sys
import subprocess
import json
import base64


def test_gfpup_run_dry_run_manifest(tmp_path):
    # Prepare a tiny valid PNG without external deps (1x1 gray pixel)
    # This is a minimal PNG file (base64) for a 1x1 pixel image.
    png_1x1_gray_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAr8B9d8wqgAAAABJRU5ErkJggg=="
    )

    inp_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    inp_dir.mkdir()
    out_dir.mkdir()

    img_path = inp_dir / "x.png"
    img_path.write_bytes(base64.b64decode(png_1x1_gray_b64))

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_path = os.path.join(repo_root, "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        "-m",
        "gfpp.cli",
        "run",
        "--input",
        str(img_path),
        "--output",
        str(out_dir),
        "--dry-run",
        "--metrics",
        "off",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    man_path = out_dir / "manifest.json"
    assert man_path.exists(), f"manifest not found: {man_path}"
    data = json.loads(man_path.read_text())
    assert "args" in data and "results" in data
    assert isinstance(data["results"], list)
