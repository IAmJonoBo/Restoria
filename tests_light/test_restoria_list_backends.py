# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import json


def test_restoria_list_backends_json():
    import sys

    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from restoria.cli.main import main

    # Capture JSON output
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(["list-backends", "--json"])  # type: ignore[arg-type]
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert isinstance(data, dict)
    assert data.get("schema_version") == "1"
    assert "backends" in data and isinstance(data["backends"], dict)
