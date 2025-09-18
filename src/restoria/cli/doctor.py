# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse


def doctor_cmd(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="restoria doctor")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    py_ver = None
    torch_ver = None
    cuda = None
    cuda_device = None
    providers = None
    suggested: list[str] = []

    # Python version
    try:
        import platform

        py_ver = platform.python_version()
    except Exception:
        py_ver = None

    # Torch/CUDA info (best-effort)
    try:
        import torch  # type: ignore

        torch_ver = getattr(torch, "__version__", None)
        cuda = bool(torch.cuda.is_available())
        if cuda:
            try:
                cuda_device = torch.cuda.get_device_name(0)
            except Exception:
                cuda_device = None
    except Exception:
        cuda = None
        torch_ver = None

    # ONNX Runtime providers (best-effort)
    try:
        import onnxruntime as ort  # type: ignore

        providers = list(getattr(ort, "get_available_providers", lambda: [])())
    except Exception:
        providers = None

    # Simple suggestions
    try:
        if cuda:
            suggested.append("--device cuda")
            # torch.compile often helps on CUDA if available
            suggested.append("--compile")
        if isinstance(providers, (list, tuple)) and any(
            p in providers
            for p in (
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
            )
        ):
            suggested.append("--backend gfpgan-ort")
    except Exception:
        pass

    payload = {
        "python": py_ver,
        "torch": torch_ver,
        "cuda_available": cuda,
        "cuda_device": cuda_device,
        "onnxruntime_providers": providers,
        "suggested_flags": suggested,
    }

    if args.json:
        try:
            import json

            print(json.dumps(payload))
        except Exception:
            print(str(payload))
        return 0

    # Text output (best-effort)
    try:
        print(f"Python: {py_ver}")
        if torch_ver is not None:
            print(f"Torch: {torch_ver}")
        print(f"CUDA available: {cuda}")
        if cuda and cuda_device:
            print(f"CUDA device: {cuda_device}")
    except Exception:
        pass
    print(f"ONNX Runtime providers: {providers}")
    if suggested:
        try:
            print("Suggested flags: " + " ".join(suggested))
        except Exception:
            pass
    return 0
