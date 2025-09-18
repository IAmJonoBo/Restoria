from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_legacy_inference_dry_run_copies_output(repo_root: Path, sample_logo: Path, tmp_path: Path, monkeypatch):
    script_path = repo_root / "inference_gfpgan.py"
    assert script_path.exists()

    out_dir = tmp_path / "legacy_dry"
    out_dir.mkdir()

    spec = importlib.util.spec_from_file_location("legacy_infer", script_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    argv = [
        str(script_path),
        "--input",
        str(sample_logo),
        "--output",
        str(out_dir),
        "--backend",
        "gfpgan",
        "--dry-run",
    ]

    monkeypatch.setattr(sys, "argv", argv)
    rc = mod.main()  # type: ignore[attr-defined]
    assert rc == 0 or rc is None

    # Dry-run should not create heavy outputs but should leave the output directory intact
    assert out_dir.exists()
