# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import json
import os
import tempfile


PNG_1x1_GRAY_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAr8B9d8wqgAAAABJRU5ErkJggg=="
)


def _repo_root() -> str:
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, os.pardir))


def _ensure_sys_path():
    import sys

    repo_root = _repo_root()
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _cleanup_tmp(path: str) -> None:
    if not os.environ.get("KEEP_TMP"):
        try:
            for fn in os.listdir(path):
                try:
                    os.remove(os.path.join(path, fn))
                except Exception:
                    pass
            os.rmdir(path)
        except Exception:
            pass


def test_restoria_normal_run_manifest_seed_and_deterministic_present():
    _ensure_sys_path()
    from restoria.cli import run as runmod
    run_cmd = runmod.run_cmd

    in_dir = tempfile.mkdtemp(prefix="restoria_in_")
    out_dir = tempfile.mkdtemp(prefix="restoria_out_")
    try:
        p = os.path.join(in_dir, "a.png")
        with open(p, "wb") as f:
            f.write(base64.b64decode(PNG_1x1_GRAY_B64))

        # Monkeypatch loader to avoid cv2 dependency in light test
        orig_loader = runmod._load_image
        runmod._load_image = lambda _p: object()

        rc = run_cmd([
            "--input", p,
            "--output", out_dir,
            "--backend", "gfpgan",
            "--metrics", "off",
            "--seed", "777",
            "--deterministic",
        ])
        assert rc == 0

        man_path = os.path.join(out_dir, "manifest.json")
        met_path = os.path.join(out_dir, "metrics.json")
        assert os.path.exists(man_path)
        assert os.path.exists(met_path)

        man = json.load(open(man_path))
        assert man.get("args", {}).get("seed") == 777
        assert man.get("args", {}).get("deterministic") is True
        assert isinstance(man.get("results"), list)
        # Should have one result record
        assert len(man.get("results")) == 1
    finally:
        try:
            # Restore original loader
            runmod._load_image = orig_loader  # type: ignore
        except Exception:
            pass
        _cleanup_tmp(in_dir)
        _cleanup_tmp(out_dir)
