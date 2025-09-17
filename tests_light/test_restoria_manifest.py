from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _write_dummy_png(path: str) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        img = (np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.imwrite(path, img)  # type: ignore[attr-defined]
        return
    except Exception:
        # Fallback: write an empty file with .png extension; dry-run path copies
        Path(path).write_bytes(b"")


def _read_manifest(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def test_restoria_plan_only_manifest_seed_and_deterministic_present():
    from restoria.cli.run import run_cmd

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.png")
        _write_dummy_png(src)
        out = os.path.join(td, "out")
        code = run_cmd([
            "--input", src,
            "--output", out,
            "--backend", "gfpgan",
            "--plan-only",
            "--seed", "123",
            "--deterministic",
        ])
        assert code == 0
        man = _read_manifest(os.path.join(out, "manifest.json"))
        assert man.get("args", {}).get("seed") == 123
        assert man.get("args", {}).get("deterministic") is True
        # Basic shape checks
        assert man.get("metrics_file") == "metrics.json"
        assert isinstance(man.get("results"), list)


def test_restoria_dry_run_manifest_seed_and_deterministic_present():
    from restoria.cli.run import run_cmd

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.png")
        _write_dummy_png(src)
        out = os.path.join(td, "out")
        code = run_cmd([
            "--input", src,
            "--output", out,
            "--backend", "gfpgan",
            "--dry-run",
            "--seed", "321",
            "--deterministic",
        ])
        assert code == 0
        man = _read_manifest(os.path.join(out, "manifest.json"))
        assert man.get("args", {}).get("seed") == 321
        assert man.get("args", {}).get("deterministic") is True
        # Basic shape checks
        assert man.get("metrics_file") == "metrics.json"
        assert isinstance(man.get("results"), list)
