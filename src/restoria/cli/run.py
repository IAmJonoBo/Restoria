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

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _save_image(path: str, img) -> None:
    try:
        import cv2  # type: ignore

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)
    except Exception:
        pass


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


def _make_restorer(args, plan):
    from ..core import registry as _registry  # lazy

    rest_cls = _registry.get(plan.backend)
    try:
        rest = rest_cls(device=args.device)
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


def _process_one(pth: str, rest, base_cfg: dict[str, Any], arc, lpips, dists, args, plan) -> dict[str, Any] | None:
    img = _load_image(pth)
    if img is None:
        _warn(f"Failed to load: {pth}")
        return None
    cfg_img = dict(base_cfg)
    cfg_img["input_path"] = pth
    try:
        res = rest.restore(img, cfg_img)
    except Exception as e:
        _warn(f"Restore failed for {pth}: {e}")
        return None
    base, _ = os.path.splitext(os.path.basename(pth))
    out_img = os.path.join(args.output, f"{base}.png")
    if res and getattr(res, "restored_image", None) is not None:
        _save_image(out_img, res.restored_image)
    else:
        try:
            shutil.copy2(pth, out_img)
        except Exception:
            pass
    metrics: dict[str, Any] = {}
    if res and isinstance(getattr(res, "metrics", None), dict):
        metrics.update(res.metrics)
    if arc is not None:
        try:
            a = arc.cosine_from_paths(pth, out_img)
            if a is not None:
                metrics["arcface_cosine"] = a
        except Exception:
            pass
    # Optional perceptual metrics for --metrics full
    if args.metrics == "full" and lpips is not None:
        try:
            d = lpips.distance_from_paths(pth, out_img)
            if isinstance(d, (int, float)):
                metrics["lpips_alex"] = d
        except Exception:
            pass
    if args.metrics == "full" and dists is not None:
        try:
            dv = dists.distance_from_paths(pth, out_img)
            if isinstance(dv, (int, float)):
                metrics["dists"] = dv
        except Exception:
            pass
    return {
        "input": pth,
        "backend": plan.backend,
        "restored_img": out_img,
        "metrics": metrics,
    }


def _run_with_registry(args, inputs: list[str]) -> int:
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
                },
                device=args.device,
                results=[],
                metrics_file=METRICS_FILENAME,
                env=runtime_env,
            )
            write_manifest(os.path.join(args.output, MANIFEST_FILENAME), man)
        except Exception:
            pass
    else:
        rest, base_cfg = _make_restorer(args, plan)
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

            # Resolve effective device if auto
            dev = args.device
            if dev == "auto":
                try:
                    import torch  # type: ignore

                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    dev = None
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
                },
                device=dev,
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
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--experimental", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--compile", action="store_true", help="Hint to enable torch.compile if available")
    p.add_argument(
        "--ort-providers",
        nargs="+",
        default=[],
        help="Preferred ONNX Runtime providers (e.g., CPUExecutionProvider CUDAExecutionProvider)",
    )
    p.add_argument("--plan-only", action="store_true", help="Compute plan and write to output without running")
    args = p.parse_args(argv)

    inputs = _list_inputs(args.input)
    if args.dry_run:
        recs: list[dict[str, Any]] = []
        for pth in inputs:
            base, _ = os.path.splitext(os.path.basename(pth))
            out_img = os.path.join(args.output, f"{base}.png")
            img = _load_image(pth)
            if img is None:
                # Fallback: file copy when image decode unavailable
                os.makedirs(os.path.dirname(out_img), exist_ok=True)
                try:
                    shutil.copy2(pth, out_img)
                except Exception:
                    _warn(f"Failed to load/copy: {pth}")
                    continue
            else:
                _save_image(out_img, img)
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
                },
                device=args.device,
                results=recs,
                metrics_file=METRICS_FILENAME,
                env=runtime_env,
            )
            write_manifest(os.path.join(args.output, MANIFEST_FILENAME), man)
        except Exception:
            pass
        return 0

    # Preferred path: use registry + orchestrator with graceful fallback
    try:
        return _run_with_registry(args, inputs)
    except Exception as e:
        _warn(f"Registry path failed, delegating to legacy CLI: {e}")
        return _delegate_legacy(args)
