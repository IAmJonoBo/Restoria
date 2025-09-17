# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from typing import Any


def list_backends_cmd(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="restoria list-backends")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    p.add_argument("--all", action="store_true", help="Include experimental backends")
    args = p.parse_args(argv)

    try:
        from ..core.registry import list_backends  # lazy

        data = list_backends(include_experimental=bool(args.all))
        if args.json:
            out: dict[str, Any] = {
                "schema_version": "1",
                "experimental": bool(args.all),
                "backends": data,
            }
            print(json.dumps(out))
        else:
            print("Available backends:")
            for k, ok in sorted(data.items()):
                mark = "✓" if ok else "✗"
                print(f"  {mark} {k}")
        return 0
    except Exception as e:
        print(f"[WARN] Failed to list backends: {e}")
        return 1
