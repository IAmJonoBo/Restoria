# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path


def test_restoria_plan_only_includes_params(sample_logo: Path, tmp_path: Path):
    from restoria.cli.run import run_cmd

    out_dir = tmp_path / "restoria_plan"
    out_dir.mkdir()

    rc = run_cmd([
        "--input",
        str(sample_logo),
        "--output",
        str(out_dir),
        "--plan-only",
        "--compile",
        "--ort-providers",
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
    ])
    assert rc == 0

    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text())
    params = data.get("plan", {}).get("params", {})
    assert params.get("compile") is True
    assert params.get("ort_providers") == ["CPUExecutionProvider", "CUDAExecutionProvider"]
