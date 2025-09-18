# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

from .run import run_cmd
from .bench import bench_cmd
from .doctor import doctor_cmd
from .list_backends import list_backends_cmd


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: restoria <run|bench|doctor|list-backends> [options]")
        return 2
    # Gracefully handle top-level help
    if argv and argv[0] in {"--help", "-h"}:
        print("Usage: restoria <run|bench|doctor|list-backends> [options]")
        print("Try: restoria run --help")
        return 0
    cmd, *rest = argv
    if cmd == "run":
        return run_cmd(rest)
    if cmd == "bench":
        return bench_cmd(rest)
    if cmd == "doctor":
        return doctor_cmd(rest)
    if cmd in {"list-backends", "list_backends"}:
        return list_backends_cmd(rest)
    print(f"Unknown command: {cmd}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
