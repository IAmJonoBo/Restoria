# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse


def doctor_cmd(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="restoria doctor")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    py_ver = None
    cuda = None
    providers = None
    try:
        import platform

        py_ver = platform.python_version()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = None
    try:
        import onnxruntime as ort  # type: ignore

        providers = list(getattr(ort, "get_available_providers", lambda: [])())
    except Exception:
        providers = None

    payload = {
        "python": py_ver,
        "cuda_available": cuda,
        "onnxruntime_providers": providers,
        "suggested_flags": [],
    }

    if args.json:
        try:
            import json

            print(json.dumps(payload))
        except Exception:
            print(str(payload))
        return 0

    print(f"Python: {py_ver}")
    print(f"CUDA available: {cuda}")
    print(f"ONNX Runtime providers: {providers}")
    return 0
