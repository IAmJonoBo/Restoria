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


def test_restoria_dry_run_folder_writes_manifest_and_metrics():
    _ensure_sys_path()
    from restoria.cli.run import run_cmd

    # Create a temp input folder with a couple of tiny PNGs
    in_dir = tempfile.mkdtemp(prefix="restoria_in_")
    out_dir = tempfile.mkdtemp(prefix="restoria_out_")
    try:
        for name in ("a.png", "b.png"):
            p = os.path.join(in_dir, name)
            with open(p, "wb") as f:
                f.write(base64.b64decode(PNG_1x1_GRAY_B64))

        rc = run_cmd([
            "--input", in_dir,
            "--output", out_dir,
            "--dry-run",
        ])
        assert rc == 0

        man_path = os.path.join(out_dir, "manifest.json")
        met_path = os.path.join(out_dir, "metrics.json")
        assert os.path.exists(man_path)
        assert os.path.exists(met_path)

        mdata = json.load(open(met_path))
        # metrics.json in dry-run contains list of processed files under "metrics"
        metrics = mdata.get("metrics", [])
        assert isinstance(metrics, list)
        assert len(metrics) == 2

        j = json.load(open(man_path))
        assert j.get("metrics_file") == "metrics.json"
        results = j.get("results")
        assert isinstance(results, list)
        assert len(results) == 2
    finally:
        _cleanup_tmp(in_dir)
        _cleanup_tmp(out_dir)
