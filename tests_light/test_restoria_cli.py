# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path


def test_restoria_dry_run_processes_sample(sample_logo: Path, tmp_path: Path):
    from restoria.cli.run import run_cmd

    out_dir = tmp_path / "restoria_dry"
    out_dir.mkdir()

    rc = run_cmd(["--input", str(sample_logo), "--output", str(out_dir), "--dry-run"])
    assert rc == 0

    out_img = out_dir / sample_logo.name
    assert out_img.exists()
    assert (out_dir / "metrics.json").exists()
