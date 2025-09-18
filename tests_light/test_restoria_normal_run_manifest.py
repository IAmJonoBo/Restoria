# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import json
from pathlib import Path


PNG_1x1_GRAY_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAr8B9d8wqgAAAABJRU5ErkJggg=="
)


def test_restoria_normal_run_manifest_seed_and_deterministic_present(tmp_path: Path):
    from restoria.cli import run as runmod

    run_cmd = runmod.run_cmd

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    sample_path = input_dir / "a.png"
    sample_path.write_bytes(base64.b64decode(PNG_1x1_GRAY_B64))

    orig_loader = runmod._load_image
    runmod._load_image = lambda _p: object()

    try:
        rc = run_cmd(
            [
                "--input",
                str(sample_path),
                "--output",
                str(output_dir),
                "--backend",
                "gfpgan",
                "--metrics",
                "off",
                "--seed",
                "777",
                "--deterministic",
            ]
        )
        assert rc == 0

        man_path = output_dir / "manifest.json"
        met_path = output_dir / "metrics.json"
        assert man_path.exists()
        assert met_path.exists()

        manifest = json.loads(man_path.read_text())
        assert manifest.get("args", {}).get("seed") == 777
        assert manifest.get("args", {}).get("deterministic") is True
        results = manifest.get("results")
        assert isinstance(results, list) and len(results) == 1
    finally:
        runmod._load_image = orig_loader  # type: ignore
