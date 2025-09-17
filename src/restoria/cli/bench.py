# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import time


def bench_cmd(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="restoria bench")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--backend",
        default="gfpgan",
        choices=["gfpgan", "gfpgan-ort", "codeformer", "restoreformerpp", "diffbir", "hypir"],
    )
    args = p.parse_args(argv)

    t0 = time.time()
    try:
        from .run import run_cmd

        rc = run_cmd(["--input", args.input, "--output", args.output, "--backend", args.backend])
    except Exception:
        rc = 1
    elapsed = time.time() - t0
    print(f"Benchmark completed in {elapsed:.2f}s (rc={rc})")
    return rc
