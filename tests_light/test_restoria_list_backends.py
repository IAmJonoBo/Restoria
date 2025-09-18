# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path


def test_restoria_list_backends_json(repo_root: Path):
    from restoria.cli.main import main

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["list-backends", "--json"])  # type: ignore[arg-type]
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert isinstance(data, dict)
    assert data.get("schema_version") == "2"
    backends = data.get("backends")
    assert isinstance(backends, dict)
    sample_backend = next(iter(backends.values()))
    assert "available" in sample_backend
    assert "metadata" in sample_backend
