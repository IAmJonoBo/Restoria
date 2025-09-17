# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import tempfile


def test_restoria_plan_only_includes_params():
    import sys

    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from restoria.cli.run import run_cmd

    sample = os.path.join(repo_root, "assets", "gfpgan_logo.png")
    assert os.path.exists(sample)

    out_dir = tempfile.mkdtemp(prefix="restoria_plan_")
    try:
        rc = run_cmd([
            "--input", sample,
            "--output", out_dir,
            "--plan-only",
            "--compile",
            "--ort-providers", "CPUExecutionProvider", "CUDAExecutionProvider",
        ])
        assert rc == 0
        mpath = os.path.join(out_dir, "metrics.json")
        assert os.path.exists(mpath)
        data = json.load(open(mpath))
        assert "plan" in data
        params = data["plan"].get("params", {})
        # compile flag should be threaded
        assert params.get("compile") is True
        # providers should be captured
        assert params.get("ort_providers") == ["CPUExecutionProvider", "CUDAExecutionProvider"]
    finally:
        if not os.environ.get("KEEP_TMP"):
            try:
                for fn in os.listdir(out_dir):
                    try:
                        os.remove(os.path.join(out_dir, fn))
                    except Exception:
                        pass
                os.rmdir(out_dir)
            except Exception:
                pass
