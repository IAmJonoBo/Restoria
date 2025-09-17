"""
Lightweight checks that Restoria writes a manifest.json in plan-only and dry-run modes.
Keeps shape checks minimal to avoid coupling to internal schema changes.
"""
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import tempfile


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


def test_restoria_plan_only_writes_manifest():
    _ensure_sys_path()
    from restoria.cli.run import run_cmd

    repo_root = _repo_root()
    sample = os.path.join(repo_root, "assets", "gfpgan_logo.png")
    assert os.path.exists(sample)

    out_dir = tempfile.mkdtemp(prefix="restoria_manifest_plan_")
    try:
        rc = run_cmd([
            "--input",
            sample,
            "--output",
            out_dir,
            "--plan-only",
        ])
        assert rc == 0
        man_path = os.path.join(out_dir, "manifest.json")
        assert os.path.exists(man_path)
        data = json.load(open(man_path))
        # Minimal shape checks
        assert isinstance(data.get("args"), dict)
        assert data.get("metrics_file") == "metrics.json"
        assert isinstance(data.get("results"), list)
    finally:
        _cleanup_tmp(out_dir)


def test_restoria_dry_run_writes_manifest():
    _ensure_sys_path()
    from restoria.cli.run import run_cmd

    repo_root = _repo_root()
    sample = os.path.join(repo_root, "assets", "gfpgan_logo.png")
    assert os.path.exists(sample)

    out_dir = tempfile.mkdtemp(prefix="restoria_manifest_dry_")
    try:
        rc = run_cmd([
            "--input",
            sample,
            "--output",
            out_dir,
            "--dry-run",
        ])
        assert rc == 0
        man_path = os.path.join(out_dir, "manifest.json")
        assert os.path.exists(man_path)
        data = json.load(open(man_path))
        assert isinstance(data.get("args"), dict)
        assert data.get("metrics_file") == "metrics.json"
        assert isinstance(data.get("results"), list)
        # device should be present for dry-run path
        assert data.get("device") is not None
    finally:
        _cleanup_tmp(out_dir)
