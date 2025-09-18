# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import json
from pathlib import Path


PNG_1x1_GRAY_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAr8B9d8wqgAAAABJRU5ErkJggg=="
)


def test_restoria_dry_run_folder_writes_manifest_and_metrics(tmp_path: Path):
    from restoria.cli.run import run_cmd

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    for name in ("a.png", "b.png"):
        (input_dir / name).write_bytes(base64.b64decode(PNG_1x1_GRAY_B64))

    rc = run_cmd([
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--dry-run",
    ])
    assert rc == 0

    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    assert manifest_path.exists()
    assert metrics_path.exists()

    metrics_payload = json.loads(metrics_path.read_text())
    metrics = metrics_payload.get("metrics", [])
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("metrics_file") == "metrics.json"
    results = manifest.get("results")
    assert isinstance(results, list)
    assert len(results) == 2
