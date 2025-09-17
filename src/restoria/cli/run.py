# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

METRICS_FILENAME = "metrics.json"
MANIFEST_FILENAME = "manifest.json"


def _warn(msg: str) -> None:
    try:
        print(f"[WARN] {msg}")
    except Exception:
        pass


def _set_deterministic(seed: int | None, deterministic: bool) -> None:
    """Best-effort deterministic setup.

    - Seeds Python's random, NumPy, and torch (if present)
    - Enables deterministic CuDNN and disables benchmark mode when requested
    """
    try:
        import random

        if seed is not None:
            random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore

        if seed is not None:
            np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if seed is not None:
            torch.manual_seed(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass


def _resolve_device(requested: str) -> str:
    """Resolve effective device string.

    Preference order when requested == 'auto': CUDA > MPS > CPU.
    Falls back to CPU on any error or when frameworks are unavailable.
    """
    if requested != "auto":
        return requested
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def _list_inputs(inp: str) -> list[str]:
    if os.path.isdir(inp):
        return [
            os.path.join(inp, p)
            for p in sorted(os.listdir(inp))
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    return [inp]


def _load_image(path: str):
    try:
        import cv2  # type: ignore
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # type: ignore[attr-defined]
        return img
    except Exception:
        return None


def _save_image(path: str, img) -> None:
    try:
        import cv2  # type: ignore

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)  # type: ignore[attr-defined]
    except Exception:
        pass


def _save_or_copy_image(out_img: str, img, src_path: str) -> None:
    if img is None:
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        try:
            shutil.copy2(src_path, out_img)
        except Exception:
            _warn(f"Failed to load/copy: {src_path}")
        return
    _save_image(out_img, img)


def _delegate_legacy(args) -> int:
    try:
        from gfpp.cli import cmd_run as _gfpp_run  # type: ignore

        mapped = [
            "--input",
            args.input,
            "--output",
            args.output,
            "--backend",
            args.backend,
            "--metrics",
            args.metrics,
            "--device",
            args.device,
        ]
        return _gfpp_run(mapped)
    except Exception as e2:
        _warn(f"Delegation to legacy CLI failed: {e2}")
        return 1


def _compute_plan(args, inputs: list[str]):
    from ..core import orchestrator as _orch  # lazy

    first_img_path = inputs[0] if inputs else None
    plan_opts = {
        "backend": args.backend,
        "experimental": bool(args.experimental),
        "compile": bool(getattr(args, "compile", False)),
        "ort_providers": list(getattr(args, "ort_providers", []) or []),
    }
    return _orch.plan(first_img_path or "", plan_opts)


def _make_restorer(args, plan, resolved_device: str | None):
    from ..core import registry as _registry  # lazy

    rest_cls = _registry.get(plan.backend)
    try:
        dev = resolved_device or args.device
        rest = rest_cls(device=dev)
    except TypeError:
        rest = rest_cls()
    base_cfg: dict[str, Any] = dict(plan.params)
    rest.prepare(base_cfg)
    return rest, base_cfg


def _maybe_arcface(args):
    if args.metrics not in {"fast", "full"}:
        return None
    try:
        from gfpp.metrics import ArcFaceIdentity  # type: ignore

        return ArcFaceIdentity()
    except Exception:
        return None


def _maybe_lpips_dists(args):
    """Optionally construct LPIPS and DISTS metrics for --metrics full.

    Returns (lpips, dists), either or both may be None if unavailable.
    """
    if args.metrics != "full":
        return None, None
    lpips = None
    dists = None
    try:
        from gfpp.metrics import LPIPSMetric  # type: ignore

        lpips = LPIPSMetric()
    except Exception:
        lpips = None
    try:
        from gfpp.metrics import DISTSMetric  # type: ignore

        dists = DISTSMetric()
    except Exception:
        dists = None
    return lpips, dists


def _add_arcface_metric(metrics: dict[str, Any], arc, pth: str, out_img: str) -> None:
    if arc is None:
        return
    try:
        a = arc.cosine_from_paths(pth, out_img)
        if a is not None:
            metrics["arcface_cosine"] = a
    except Exception:
        pass


def _add_lpips_metric(metrics: dict[str, Any], lpips, enabled: bool, pth: str, out_img: str) -> None:
    if not enabled or lpips is None:
        return
    try:
        d = lpips.distance_from_paths(pth, out_img)
        if isinstance(d, (int, float)):
            metrics["lpips_alex"] = d
    except Exception:
        pass


def _add_dists_metric(metrics: dict[str, Any], dists, enabled: bool, pth: str, out_img: str) -> None:
    if not enabled or dists is None:
        return
    try:
        dv = dists.distance_from_paths(pth, out_img)
        if isinstance(dv, (int, float)):
            metrics["dists"] = dv
    except Exception:
        pass


def _compute_metrics(pth: str, out_img: str, res, args, arc, lpips, dists) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if res and isinstance(getattr(res, "metrics", None), dict):
        metrics.update(res.metrics)
    _add_arcface_metric(metrics, arc, pth, out_img)
    is_full = args.metrics == "full"
    _add_lpips_metric(metrics, lpips, is_full, pth, out_img)
    _add_dists_metric(metrics, dists, is_full, pth, out_img)
    return metrics


def _process_one(pth: str, rest, base_cfg: dict[str, Any], arc, lpips, dists, args, plan) -> dict[str, Any] | None:
    img = _load_image(pth)
    base, _ = os.path.splitext(os.path.basename(pth))
    out_img = os.path.join(args.output, f"{base}.png")

    # If image failed to load, fallback to copying source and record minimal result
    if img is None:
        _warn(f"Failed to load: {pth}")
        _save_or_copy_image(out_img, None, pth)  # force copy
        metrics = _compute_metrics(pth, out_img, None, args, arc, lpips, dists)
        return {
            "input": pth,
            "backend": plan.backend,
            "restored_img": out_img,
            "metrics": metrics,
        }

    cfg_img = dict(base_cfg)
    cfg_img["input_path"] = pth
    res = None
    try:
        res = rest.restore(img, cfg_img)
    except Exception as e:
        # Graceful fallback: copy source and continue with empty metrics
        _warn(f"Restore failed for {pth}: {e}")
        _save_or_copy_image(out_img, None, pth)  # force copy to avoid cv2 dependency
        metrics = _compute_metrics(pth, out_img, None, args, arc, lpips, dists)
        return {
            "input": pth,
            "backend": plan.backend,
            "restored_img": out_img,
            "metrics": metrics,
        }

    if res and getattr(res, "restored_image", None) is not None:
        _save_image(out_img, res.restored_image)
    else:
        # No restored image produced, copy source as best-effort
        _save_or_copy_image(out_img, None, pth)
    metrics = _compute_metrics(pth, out_img, res, args, arc, lpips, dists)
    return {
        "input": pth,
        "backend": plan.backend,
        "restored_img": out_img,
        "metrics": metrics,
    }


def _do_dry_run(args, inputs: list[str], resolved_device: str) -> int:
    recs: list[dict[str, Any]] = []
    for pth in inputs:
        base, _ = os.path.splitext(os.path.basename(pth))
        out_img = os.path.join(args.output, f"{base}.png")
        img = _load_image(pth)
        _save_or_copy_image(out_img, img, pth)
        recs.append({"input": pth, "restored_img": out_img})
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, METRICS_FILENAME), "w") as f:
        json.dump({"metrics": recs}, f, indent=2)
    print(f"Processed {len(recs)} files (dry-run) -> {args.output}")
    # Write minimal manifest for dry-run
    try:
        from ..io.manifest import RunManifest, write_manifest  # type: ignore

        runtime_env = {
            "runtime": {
                "compile": bool(getattr(args, "compile", False)),
                "ort_providers": list(getattr(args, "ort_providers", []) or []),
            }
        }
        man = RunManifest(
            args={
                "backend": args.backend,
                "metrics": args.metrics,
                "device": args.device,
                "experimental": bool(args.experimental),
                "dry_run": True,
                "seed": args.seed,
                "deterministic": bool(args.deterministic),
            },
            device=resolved_device,
            results=recs,
            metrics_file=METRICS_FILENAME,
            env=runtime_env,
        )
        write_manifest(os.path.join(args.output, MANIFEST_FILENAME), man)
    except Exception:
        pass
    return 0


def _run_with_registry(args, inputs: list[str], resolved_device: str | None) -> int:
    os.makedirs(args.output, exist_ok=True)
    plan = _compute_plan(args, inputs)
    rc = 0
    if getattr(args, "plan_only", False):
        with open(os.path.join(args.output, METRICS_FILENAME), "w") as f:
            json.dump({
                "metrics": [],
                "plan": {
                    "backend": plan.backend,
                    "reason": plan.reason,
                    "params": plan.params,
                },
            }, f, indent=2)
        print(f"Wrote plan only -> {args.output}")
        # Also write a lightweight manifest with runtime hints
        try:
            from ..io.manifest import RunManifest, write_manifest  # type: ignore

            runtime_env = {
                "runtime": {
                    "compile": bool(getattr(args, "compile", False)),
                    "ort_providers": list(getattr(args, "ort_providers", []) or []),
                }
            }
            man = RunManifest(
                args={
                    "backend": args.backend,
                    "metrics": args.metrics,
                    "device": args.device,
                    "experimental": bool(args.experimental),
                    "plan_only": True,
                    "seed": args.seed,
                    "deterministic": bool(args.deterministic),
                },
                device=resolved_device or args.device,
                results=[],
                metrics_file=METRICS_FILENAME,
                env=runtime_env,
            )
            write_manifest(os.path.join(args.output, MANIFEST_FILENAME), man)
        except Exception:
            pass
    else:
        rest, base_cfg = _make_restorer(args, plan, resolved_device)
        arc = _maybe_arcface(args)
        lp, ds = _maybe_lpips_dists(args)
        recs: list[dict[str, Any]] = []
        for pth in inputs:
            rec = _process_one(pth, rest, base_cfg, arc, lp, ds, args, plan)
            if rec is not None:
                recs.append(rec)
        with open(os.path.join(args.output, METRICS_FILENAME), "w") as f:
            json.dump({
                "metrics": recs,
                "plan": {
                    "backend": plan.backend,
                    "reason": plan.reason,
                    "params": getattr(plan, "params", {}),
                },
            }, f, indent=2)
        print(f"Processed {len(recs)} files -> {args.output}")
        # Write manifest with runtime info (best-effort)
        try:
            from ..io.manifest import RunManifest, write_manifest  # type: ignore

            runtime_env = {
                "runtime": {
                    "compile": bool(getattr(args, "compile", False)),
                    "ort_providers": list(getattr(args, "ort_providers", []) or []),
                }
            }
            man = RunManifest(
                args={
                    "backend": args.backend,
                    "metrics": args.metrics,
                    "device": args.device,
                    "experimental": bool(args.experimental),
                    "dry_run": False,
                    "seed": args.seed,
                    "deterministic": bool(args.deterministic),
                },
                device=resolved_device,
                results=recs,
                metrics_file=METRICS_FILENAME,
                env=runtime_env,
            )
            write_manifest(os.path.join(args.output, MANIFEST_FILENAME), man)
        except Exception:
            pass
    return rc


def run_cmd(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="restoria run")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--backend",
        default="gfpgan",
        choices=["gfpgan", "gfpgan-ort", "codeformer", "restoreformerpp", "diffbir", "hypir"],
    )
    p.add_argument("--metrics", default="off", choices=["off", "fast", "full"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--experimental", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic backend behavior when possible")
    p.add_argument("--compile", action="store_true", help="Hint to enable torch.compile if available")
    p.add_argument(
        "--ort-providers",
        nargs="+",
        default=[],
        help="Preferred ONNX Runtime providers (e.g., CPUExecutionProvider CUDAExecutionProvider)",
    )
    p.add_argument("--plan-only", action="store_true", help="Compute plan and write to output without running")
    args = p.parse_args(argv)

    # Best-effort determinism setup before any heavy imports/initialization
    _set_deterministic(args.seed, bool(args.deterministic))

    inputs = _list_inputs(args.input)
    resolved_device = _resolve_device(args.device)
    if args.dry_run:
        return _do_dry_run(args, inputs, resolved_device)

    # Preferred path: use registry + orchestrator with graceful fallback
    try:
        return _run_with_registry(args, inputs, resolved_device)
    except Exception as e:
        _warn(f"Registry path failed, delegating to legacy CLI: {e}")
        return _delegate_legacy(args)
