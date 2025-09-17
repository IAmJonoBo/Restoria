# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tempfile


def test_restoria_dry_run_processes_sample():
    # Import locally from src if not installed
    import sys

    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from restoria.cli.run import run_cmd

    sample = os.path.join(repo_root, "assets", "gfpgan_logo.png")
    assert os.path.exists(sample)

    out_dir = tempfile.mkdtemp(prefix="restoria_dry_")
    try:
        rc = run_cmd(["--input", sample, "--output", out_dir, "--dry-run"])
        assert rc == 0
        # Output should contain a PNG named by the base of input
        out_img = os.path.join(out_dir, "gfpgan_logo.png")
        assert os.path.exists(out_img)
        # Metrics should be written
        assert os.path.exists(os.path.join(out_dir, "metrics.json"))
    finally:
        # Leave artifacts for CI inspection only if an env var is set
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
