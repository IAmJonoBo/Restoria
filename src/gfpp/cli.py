from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict
import os as _os

from .background import build_realesrgan
from .io import RunManifest, list_inputs, load_image_bgr, save_image, write_manifest
from .metrics import ArcFaceIdentity, DISTSMetric, LPIPSMetric
from .restorers.codeformer import CodeFormerRestorer
from .restorers.gfpgan import GFPGANRestorer
from .restorers.restoreformerpp import RestoreFormerPP
from .presets import apply_preset


# ------------------------- Small helpers (no behavior change) -------------------------
TMP_IN = "in.png"
TMP_OUT = "out.png"

_DEPRECATION_EMITTED = False

def _maybe_deprecation_notice():
    global _DEPRECATION_EMITTED
    if _DEPRECATION_EMITTED:
        return
    if _os.environ.get("GFPP_NO_DEPRECATION"):
        _DEPRECATION_EMITTED = True
        return
    try:
        import sys as _sys
        print(
            "[WARN] gfpup is deprecated; please use 'restoria' CLI (restoria run/doctor/list-backends).",
            file=_sys.stderr,
        )
    except Exception:
        pass
    _DEPRECATION_EMITTED = True


def _warn(msg: str) -> None:
    """Lightweight warning helper.

    - Tries standard logging (logger name: 'gfpp.cli')
    - Always prints with [WARN] to preserve current CLI behavior
    """
    try:
        import logging

        logging.getLogger("gfpp.cli").warning(msg)
    except Exception:
        pass
    try:
        print(f"[WARN] {msg}")
    except Exception:
        pass


def _build_bg_upsampler(background: str, quality: str, device: str):
    if background != "realesrgan":
        return None
    if quality == "quick":
        tile = 0
        prec = "fp16"
    elif quality == "best":
        tile = 0
        prec = "fp32"
    else:
        tile = 400
        prec = "auto"
    return build_realesrgan(device=device, tile=tile, precision=prec)


def _instantiate_restorer(name: str, args, bg):
    # Ensure name normalization from orchestrator
    if name == "gfpgan":
        return GFPGANRestorer(device=args.device, bg_upsampler=bg, compile_mode=args.compile), "gfpgan"
    if name == "gfpgan-ort":
        from .restorers.gfpgan_ort import ORTGFPGANRestorer

        return ORTGFPGANRestorer(device=args.device, bg_upsampler=bg), "gfpgan-ort"
    if name == "codeformer":
        return CodeFormerRestorer(device=args.device, bg_upsampler=bg), "codeformer"
    if name in {"restoreformer", "restoreformerpp"}:
        return RestoreFormerPP(device=args.device, bg_upsampler=bg), "restoreformerpp"
    if name == "hypir":
        if not getattr(args, "experimental", False):
            _warn("HYPIR is experimental. Enable with --experimental; falling back to gfpgan")
            return GFPGANRestorer(device=args.device, bg_upsampler=bg, compile_mode=args.compile), "gfpgan"
        try:
            from .restorers.hypir import HYPIRRestorer

            return HYPIRRestorer(device=args.device, bg_upsampler=bg), "hypir"
        except Exception:
            _warn(
                "HYPIR backend unavailable. Install extras: "
                "pip install -e \".[hypir]\"; falling back to gfpgan"
            )
            return GFPGANRestorer(device=args.device, bg_upsampler=bg, compile_mode=args.compile), "gfpgan"
    if name in {"ensemble", "guided"}:
        try:
            from gfpp.core.registry import get as _get_backend  # type: ignore
            backend_cls = _get_backend(name)
            return backend_cls(device=args.device, bg_upsampler=bg), name
        except Exception:
            _warn(f"Backend {name} unavailable; falling back to gfpgan")
            return GFPGANRestorer(device=args.device, bg_upsampler=bg, compile_mode=args.compile), "gfpgan"
    _warn(f"Backend {name} not yet implemented; falling back to gfpgan")
    return GFPGANRestorer(device=args.device, bg_upsampler=bg, compile_mode=args.compile), "gfpgan"


def _base_cfg(args, preset_weight: float) -> Dict[str, Any]:
    return {
        "version": args.version,
        "upscale": 2,
        "use_parse": True,
        "detector": "retinaface_resnet50",
        "weight": preset_weight,
        "no_download": args.no_download,
    }


def _apply_backend_cfg(chosen_backend: str, args, cfg: Dict[str, Any]) -> None:
    if chosen_backend == "gfpgan-ort" and args.model_path_onnx:
        cfg["model_path_onnx"] = args.model_path_onnx
    if chosen_backend == "codeformer" and args.codeformer_fidelity is not None:
        try:
            cfg["weight"] = float(args.codeformer_fidelity)
        except Exception:
            pass
    if chosen_backend == "hypir":
        if args.texture_richness is not None:
            try:
                tr = max(0.0, min(1.0, float(args.texture_richness)))
                cfg["texture_richness"] = tr
            except Exception:
                pass
        if args.prompt is not None:
            cfg["prompt"] = str(args.prompt)
        cfg["identity_lock"] = bool(args.identity_lock)


def _optimize_weight_if_requested(rest, img, cfg, preset_weight, args, arc, lpips):
    chosen_weight = cfg.get("weight")
    if not args.optimize:
        return rest.restore(img, cfg), chosen_weight
    # Parse candidate weights, keep within [0,1]
    try:
        cand = [min(max(float(x.strip()), 0.0), 1.0) for x in args.weights_cand.split(",") if x.strip()]
    except Exception:
        cand = [0.3, 0.5, 0.7]
    best_score = None
    best_res = None
    best_w = None
    for w in cand:
        cfg["weight"] = w
        r_try = rest.restore(img, cfg)
        # Score: prefer ArcFace, else -LPIPS, else proximity to preset
        score = None
        if args.metrics in {"fast", "full"} and arc and arc.available():
            import tempfile
            import cv2

            td = tempfile.mkdtemp()
            a = os.path.join(td, TMP_IN)
            b = os.path.join(td, TMP_OUT)
            cv2.imwrite(a, img)
            if r_try and r_try.restored_image is not None:
                cv2.imwrite(b, r_try.restored_image)
            else:
                b = a
            s = arc.cosine_from_paths(a, b)
            score = s if s is not None else None
        if score is None and args.metrics == "full" and lpips and lpips.available():
            import tempfile
            import cv2

            td = tempfile.mkdtemp()
            a = os.path.join(td, TMP_IN)
            b = os.path.join(td, TMP_OUT)
            cv2.imwrite(a, img)
            if r_try and r_try.restored_image is not None:
                cv2.imwrite(b, r_try.restored_image)
            else:
                b = a
            d = lpips.distance_from_paths(a, b)
            score = -d if isinstance(d, float) else None
        if score is None:
            score = -abs(w - preset_weight)
        if best_score is None or score > best_score:
            best_score = score
            best_res = r_try
            best_w = w
    cfg["weight"] = best_w
    return best_res, best_w


def _maybe_identity_retry(chosen_backend, rest, img, cfg, preset_weight, arc, args, rec):
    import tempfile
    import cv2

    td = tempfile.mkdtemp()
    a = os.path.join(td, TMP_IN)
    b = os.path.join(td, TMP_OUT)
    cv2.imwrite(a, img)
    cv2.imwrite(b, rec.get("_tmp_restored") or img)
    s0 = arc.cosine_from_paths(a, b)
    if s0 is None or s0 >= args.identity_threshold:
        return None
    cfg_strict = dict(cfg)
    if chosen_backend == "hypir":
        tr0 = cfg_strict.get("texture_richness", 0.6)
        try:
            tr0 = float(tr0)
        except Exception:
            tr0 = 0.6
        cfg_strict["texture_richness"] = max(0.2, min(1.0, tr0 - 0.2))
        cfg_strict["identity_lock"] = True
    else:
        stricter = max(0.2, float(cfg.get("weight", preset_weight)) - 0.2)
        cfg_strict["weight"] = stricter
    r2 = rest.restore(img, cfg_strict)
    cv2.imwrite(b, r2.restored_image if (r2 and r2.restored_image is not None) else img)
    s1 = arc.cosine_from_paths(a, b)
    if s1 is not None and s1 > s0:
        rec["metrics"]["identity_retry"] = True
        if chosen_backend == "hypir":
            rec["metrics"]["texture_richness"] = cfg_strict.get("texture_richness")
            rec["metrics"]["identity_lock"] = True
        else:
            rec["metrics"]["weight"] = cfg_strict.get("weight")
        return r2
    return None


def _set_deterministic(seed: int | None, deterministic: bool) -> None:
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


def cmd_run(argv: list[str]) -> int:
    # cmd entry

    p = argparse.ArgumentParser(prog="gfpup run")
    p.add_argument("--input", required=True)
    p.add_argument(
        "--backend",
        default="gfpgan",
    choices=["gfpgan", "gfpgan-ort", "codeformer", "restoreformerpp", "diffbir", "hypir", "ensemble", "guided"],
    )
    p.add_argument("--background", default="realesrgan", choices=["realesrgan", "swinir", "none"])
    p.add_argument("--preset", default="natural", choices=["natural", "detail", "document"])
    p.add_argument("--compile", default="none", choices=["none", "default", "max"])
    p.add_argument("--engine-params", default=None, help="JSON string for engine params")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--experimental", action="store_true", help="Enable experimental backends and rules")
    p.add_argument("--metrics", default="off", choices=["off", "fast", "full"])
    p.add_argument("--csv-out", default=None, help="Write metrics CSV to this path")
    p.add_argument("--html-report", default=None, help="Write HTML report to this path")
    p.add_argument("--auto-backend", action="store_true", help="Select backend per-image using quality heuristics")
    p.add_argument("--auto", action="store_true", help="Alias for --auto-backend")
    p.add_argument(
        "--dry-run", action="store_true", help="Simulate run without loading models (copy inputs to outputs)"
    )
    p.add_argument(
        "--quality", default="balanced", choices=["quick", "balanced", "best"], help="Quality vs speed preset"
    )
    p.add_argument("--identity-lock", action="store_true", help="Retry with stricter preset if identity drops")
    p.add_argument("--identity-threshold", type=float, default=0.25)
    p.add_argument("--collect-feedback", action="store_true", help="Opt-in: log per-image feedback to feedback.jsonl")
    p.add_argument("--optimize", action="store_true", help="Try multiple weights and pick best by metric")
    p.add_argument("--weights-cand", default="0.3,0.5,0.7", help="Comma-separated candidate weights for --optimize")
    p.add_argument("--video", action="store_true")
    p.add_argument("--temporal", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--version", default="1.4")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--model-path-onnx", default=None, help="Path to ONNX model (for gfpgan-ort backend)")
    p.add_argument("--codeformer-fidelity", type=float, default=None, help="CodeFormer fidelity (0..1)")
    # Ensemble-specific optional args
    p.add_argument("--ensemble-backends", default=None, help="Comma-separated backends for ensemble")
    p.add_argument("--ensemble-weights", default=None, help="Comma-separated weights for ensemble")
    # Guided-specific optional args
    p.add_argument("--reference", default=None, help="Path to reference image for guided restoration")
    # HYPIR-specific optional args (ignored by other backends)
    p.add_argument("--prompt", default=None, help="HYPIR: prompt for light text-guided biasing")
    p.add_argument("--texture-richness", type=float, default=None, help="HYPIR: texture richness (0..1)")
    args = p.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    _set_deterministic(args.seed, args.deterministic)

    # Background upsampler
    bg = _build_bg_upsampler(args.background, args.quality, args.device)

    # Support --auto alias
    if args.auto:
        args.auto_backend = True
    # Choose restorer (optionally auto per-file)
    # Initial backend/restorer
    chosen_backend = "gfpgan" if args.auto_backend else args.backend
    rest, chosen_backend = _instantiate_restorer(chosen_backend, args, bg)
    # Presets (small, testable defaults)
    preset = args.preset
    preset_weight = {"natural": 0.5, "detail": 0.7, "document": 0.3}.get(preset, 0.5)

    cfg: Dict[str, Any] = _base_cfg(args, preset_weight)
    # Apply optional preset adjustments (non-destructive)
    cfg = apply_preset(preset, cfg)
    # Ensemble/guided pass-through config keys
    if args.ensemble_backends:
        cfg["ensemble_backends"] = args.ensemble_backends
    if args.ensemble_weights:
        cfg["ensemble_weights"] = args.ensemble_weights
    if args.reference:
        cfg["reference"] = args.reference
    _apply_backend_cfg(chosen_backend, args, cfg)
    # Pass feedback flag to downstream consumers
    if getattr(args, "collect_feedback", False):
        cfg["collect_feedback"] = True

    inputs = list_inputs(args.input)
    # Optional orchestrator (new path). If import fails, we silently fall back.
    use_orchestrator = bool(args.auto_backend)
    if use_orchestrator:
        try:
            from gfpp.core.orchestrator import plan as make_plan  # type: ignore
        except Exception:
            use_orchestrator = False
    results = []

    # Dry-run path: copy inputs to outputs and optionally write metrics/report
    if args.dry_run:
        for pth in inputs:
            t0 = time.time()
            img = load_image_bgr(pth)
            if img is None:
                continue
            base, _ = os.path.splitext(os.path.basename(pth))
            out_img = os.path.join(args.output, f"{base}.png")
            save_image(out_img, img)
            rec = {"input": pth, "restored_img": out_img, "metrics": {"runtime_sec": time.time() - t0}}
            # Even in dry-run we can compute no-ref metrics (they don't require restored output)
            if args.metrics != "off":
                try:
                    from gfpp.metrics import NoRefQuality  # type: ignore
                    noref = NoRefQuality()
                    for k, v in noref.score(out_img).items():
                        rec["metrics"][k] = v
                except Exception:
                    pass
                # Optional advanced IQA (proxy) metrics
                try:
                    from gfpp.metrics.adv_quality import advanced_scores  # type: ignore

                    adv = advanced_scores(out_img)
                    for k, v in adv.items():
                        rec["metrics"][k] = v
                except Exception:
                    pass
            results.append(rec)

        # Write manifest + optional CSV/HTML
        metrics_file = os.path.join(args.output, "metrics.json") if args.metrics != "off" else None
        if metrics_file:
            with open(metrics_file, "w") as f:
                json.dump({"metrics": results}, f, indent=2)
        man = RunManifest(args=vars(args), device=args.device, results=results, metrics_file=metrics_file)
        # Attach runtime details in dry-run as well
        runtime_info = {
            "compile_mode": args.compile,
            "auto_backend": bool(args.auto or args.auto_backend),
            "collect_feedback": bool(getattr(args, "collect_feedback", False)),
        }
        try:
            import onnxruntime as _ort  # type: ignore

            runtime_info["ort_providers"] = list(getattr(_ort, "get_available_providers", lambda: [])())
        except Exception:
            runtime_info["ort_providers"] = None
        try:
            man.env["runtime"] = runtime_info
        except Exception:
            pass
        try:
            import time as _time

            man.ended_at = _time.time()
        except Exception:
            pass
        write_manifest(os.path.join(args.output, "manifest.json"), man)
        if args.csv_out:
            import csv

            keys = sorted({k for r in results for k in (r.get("metrics") or {}).keys()})
            with open(args.csv_out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["input", "restored_img", *keys])
                for r in results:
                    w.writerow([r.get("input"), r.get("restored_img"), *[r.get("metrics", {}).get(k) for k in keys]])
        if args.html_report:
            try:
                from .reports.html import write_html_report

                keys = sorted({k for r in results for k in (r.get("metrics") or {}).keys()})
                write_html_report(args.output, results, keys, args.html_report)
            except Exception:
                pass
        print(f"Processed {len(results)} files (dry-run) -> {args.output}")
        return 0

    # Metrics (optional)
    arc = ArcFaceIdentity(no_download=args.no_download) if args.metrics in {"fast", "full"} else None
    lpips = LPIPSMetric() if args.metrics == "full" else None
    dists = DISTSMetric() if args.metrics == "full" else None

    for pth in inputs:
        img = load_image_bgr(pth)
        if img is None:
            _warn(f"Failed to load: {pth}")
            continue
        cfg["input_path"] = pth
        # If orchestrator active, create a plan
        if use_orchestrator:
            try:
                pl = make_plan(
                    pth,
                    {
                        "backend": chosen_backend,
                        "background": args.background,
                        "weight": cfg.get("weight", 0.5),
                        "experimental": bool(args.experimental),
                        "prompt": args.prompt,
                    },
                )
                prev_backend = chosen_backend
                chosen_backend = pl.backend
                # merge params with precedence to plan
                for k, v in pl.params.items():
                    cfg[k] = v
                # If backend changed, re-instantiate rest accordingly (auto mode is opt-in)
                if chosen_backend != prev_backend:
                    rest, chosen_backend = _instantiate_restorer(chosen_backend, args, bg)
                plan_info = {
                    "backend": pl.backend,
                    "params": pl.params,
                    "postproc": pl.postproc,
                    "reason": pl.reason,
                    "confidence": getattr(pl, "confidence", None),
                    "quality": getattr(pl, "quality", None),
                    "faces": getattr(pl, "faces", None),
                    "detail": getattr(pl, "detail", None),
                }
            except Exception:
                plan_info = None

        vram_mb = None
        t0 = time.time()
        # Reset peak memory, if possible
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # type: ignore
        except Exception:
            pass

        # Optional auto backend per-image
        if args.auto_backend and not use_orchestrator:
            try:
                from gfpgan.auto.engine_selector import select_engine_for_image  # type: ignore

                decision = select_engine_for_image(pth)
                bname = decision.engine
            except Exception:
                bname = "gfpgan"
            if bname != chosen_backend:
                if bname == "gfpgan":
                    rest = GFPGANRestorer(device=args.device, bg_upsampler=bg, compile_mode=args.compile)
                elif bname == "codeformer":
                    rest = CodeFormerRestorer(device=args.device, bg_upsampler=bg)
                elif bname in {"restoreformer", "restoreformerpp"}:
                    rest = RestoreFormerPP(device=args.device, bg_upsampler=bg)

        # Optional multi-try optimizer
        res, chosen_weight = _optimize_weight_if_requested(rest, img, cfg, preset_weight, args, arc, lpips)
        dur = time.time() - t0
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                vram_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))  # type: ignore
        except Exception:
            vram_mb = None

        base, _ = os.path.splitext(os.path.basename(pth))
        out_img = os.path.join(args.output, f"{base}.png")
        if res and res.restored_image is not None:
            save_image(out_img, res.restored_image)
        else:
            save_image(out_img, img)

        rec = {
            "input": pth,
            "restored_img": out_img,
            "metrics": {},
        }
        if use_orchestrator:
            # plan_info already a dict from dataclass .__dict__ in caller; keep as-is
            # Backward compatibility: ensure basic keys exist
            if isinstance(plan_info, dict):
                rec["plan"] = plan_info
            else:
                rec["plan"] = {"backend": chosen_backend, "reason": "plan_object_unexpected"}
        if chosen_weight is not None:
            rec["metrics"]["weight"] = chosen_weight
        # Identity lock: if identity below threshold and ArcFace available, retry with stricter preset
        if args.identity_lock and (args.metrics in {"fast", "full"}) and arc and arc.available():
            # keep temp restored to measure s0 inside helper
            rec["_tmp_restored"] = res.restored_image if (res and res.restored_image is not None) else img
            r2 = _maybe_identity_retry(chosen_backend, rest, img, cfg, preset_weight, arc, args, rec)
            if r2 is not None:
                res = r2
                save_image(out_img, res.restored_image if res.restored_image is not None else img)
            rec.pop("_tmp_restored", None)
        # Compute metrics vs input/restored when possible
        if args.metrics != "off":
            if arc and arc.available():
                rec["metrics"]["arcface_cosine"] = arc.cosine_from_paths(pth, out_img)
            if lpips and lpips.available():
                rec["metrics"]["lpips_alex"] = lpips.distance_from_paths(pth, out_img)
            if dists and dists.available():
                rec["metrics"]["dists"] = dists.distance_from_paths(pth, out_img)
            # Add NIQE/BRISQUE (no-ref) metrics if available
            try:
                from gfpp.metrics import NoRefQuality
                noref = NoRefQuality()
                noref_scores = noref.score(out_img)
                for k, v in noref_scores.items():
                    rec["metrics"][k] = v
            except Exception:
                pass
            # Optional advanced IQA metrics
            try:
                from gfpp.metrics.adv_quality import advanced_scores  # type: ignore

                adv = advanced_scores(out_img)
                for k, v in adv.items():
                    rec["metrics"][k] = v
            except Exception:
                pass
        rec["metrics"]["runtime_sec"] = dur
        if vram_mb is not None:
            rec["metrics"]["vram_mb"] = vram_mb
        # Attach model info when available
        if res and res.metrics:
            rec["metrics"].update({k: v for k, v in res.metrics.items() if v is not None})
        results.append(rec)

    # Write manifest + metrics.json
    metrics_file = os.path.join(args.output, "metrics.json") if args.metrics != "off" else None
    if metrics_file:
        with open(metrics_file, "w") as f:
            json.dump({"metrics": results}, f, indent=2)

    man = RunManifest(args=vars(args), device=args.device, results=results, metrics_file=metrics_file)
    # Attach runtime details: compile mode and ORT providers (best-effort)
    runtime_info = {
        "compile_mode": args.compile,
        "auto_backend": bool(args.auto or args.auto_backend),
        "collect_feedback": bool(getattr(args, "collect_feedback", False)),
    }
    try:
        import onnxruntime as _ort  # type: ignore

        runtime_info["ort_providers"] = list(getattr(_ort, "get_available_providers", lambda: [])())
    except Exception:
        runtime_info["ort_providers"] = None
    try:
        man.env["runtime"] = runtime_info
    except Exception:
        pass
    # Mark end time
    try:
        import time as _time

        man.ended_at = _time.time()
    except Exception:
        pass
    write_manifest(os.path.join(args.output, "manifest.json"), man)

    # Optional CSV + HTML outputs
    if args.csv_out:
        import csv

        keys = sorted({k for r in results for k in (r.get("metrics") or {}).keys()})
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["input", "restored_img", *keys])
            for r in results:
                w.writerow([r.get("input"), r.get("restored_img"), *[r.get("metrics", {}).get(k) for k in keys]])

    if args.html_report:
        try:
            from .reports.html import write_html_report

            keys = sorted({k for r in results for k in (r.get("metrics") or {}).keys()})
            write_html_report(args.output, results, keys, args.html_report)
        except Exception:
            pass

    print(f"Processed {len(results)} files -> {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - central CLI dispatcher kept monolithic for startup cost
    _maybe_deprecation_notice()
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: gfpup run --input <path> --output out/ [options]")
        return 2
    cmd = argv[0]
    if cmd == "list-backends":
        # Lightweight listing of registered backends (no heavy imports)
        import argparse
        from gfpp.core.registry import list_backends  # type: ignore

        p = argparse.ArgumentParser(prog="gfpup list-backends")
        p.add_argument("--all", action="store_true", help="Include experimental backends")
        p.add_argument("--verbose", action="store_true", help="Show availability status")
        p.add_argument("--json", action="store_true", help="Output machine-readable JSON")
        args = p.parse_args(argv[1:])

        avail = list_backends(include_experimental=bool(args.all))
        if args.json:
            payload = {
                "schema_version": "1",
                "experimental": bool(args.all),
                "backends": avail,
            }
            try:
                print(json.dumps(payload))
            except (TypeError, ValueError):
                # Best-effort fallback to string repr
                print(str(payload))
        else:
            header = f"Backends (experimental={'on' if args.all else 'off'}):"
            print(header)
            for name, ok in avail.items():
                if args.verbose:
                    print(f"  - {name}: {'available' if ok else 'missing'}")
                else:
                    print(f"  - {name}")
        return 0
    if cmd == "run":
        return cmd_run(argv[1:])
    if cmd == "doctor":
        # Support optional JSON output while preserving text mode behavior
        import argparse as _argparse
        p = _argparse.ArgumentParser(prog="gfpup doctor")
        p.add_argument("--json", action="store_true", help="Output machine-readable JSON")
        _args = p.parse_args(argv[1:])

        py_ver = None
        torch_ver = None
        cuda_avail = None
        cuda_device = None
        providers = None
        backends = None
        suggested = []

        # Collect environment info (best-effort)
        try:
            import platform as _platform
            py_ver = _platform.python_version()
        except Exception:
            py_ver = None
        try:
            import torch as _torch  # type: ignore
            torch_ver = getattr(_torch, "__version__", None)
            cuda_avail = bool(_torch.cuda.is_available())
            if _torch.cuda.is_available():
                try:
                    cuda_device = _torch.cuda.get_device_name(0)
                except Exception:
                    cuda_device = None
        except Exception:
            pass
        try:
            import onnxruntime as _ort  # type: ignore
            providers = list(getattr(_ort, "get_available_providers", lambda: [])())
        except Exception:
            providers = None
        try:
            from gfpp.core.registry import list_backends as _list_backends  # type: ignore
            backends = _list_backends(include_experimental=True)
        except Exception:
            backends = None

        # Minimal suggestions (non-fatal)
        try:
            if cuda_avail:
                suggested.append("--device cuda")
                suggested.append("--compile default")
            if isinstance(providers, (list, tuple)) and any(p in providers for p in (
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
            )):
                suggested.append("--backend gfpgan-ort")
        except Exception:
            pass

        if _args.json:
            payload = {
                "schema_version": "1",
                "python": py_ver,
                "torch": torch_ver,
                "cuda_available": cuda_avail,
                "cuda_device": cuda_device,
                "onnxruntime_providers": providers,
                "backends": backends,
                "suggested_flags": suggested,
            }
            try:
                print(json.dumps(payload))
            except (TypeError, ValueError):
                print(str(payload))
            return 0

        # Text mode (unchanged semantics)
        try:
            import platform
            import torch  # type: ignore
            print(f"Python: {py_ver or platform.python_version()}")
            print(f"Torch: {torch_ver or getattr(torch, '__version__', None)}")
            ca = cuda_avail if cuda_avail is not None else torch.cuda.is_available()
            print(f"CUDA available: {ca}")
            if ca:
                try:
                    print(f"CUDA device: {cuda_device or torch.cuda.get_device_name(0)}")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if providers is None:
                import onnxruntime as ort  # type: ignore
                providers = getattr(ort, 'get_available_providers', lambda: [])()
            print(f"ONNX Runtime providers: {providers}")
        except Exception:
            providers = None
            print("ONNX Runtime: not installed")
        try:
            if backends is None:
                from gfpp.core.registry import list_backends  # type: ignore
                backends = list_backends(include_experimental=True)
            print("Backends:")
            for k, v in (backends or {}).items():
                print(f"  - {k}: {'available' if v else 'missing'}")
        except Exception:
            pass
        # Suggestions
        try:
            if suggested:
                print("Suggested flags: " + " ".join(suggested))
        except Exception:
            pass
        return 0
    if cmd == "export-onnx":
        # Minimal stub to document export path without heavy operations
        import argparse
        from .export import export_gfpgan_onnx
        p = argparse.ArgumentParser(prog="gfpup export-onnx")
        p.add_argument("--version", default="1.4")
        p.add_argument("--model-path", default=None)
        # Support both --out and --output for ergonomics/backward-compat
        p.add_argument("--out", dest="out", default="gfpgan.onnx")
        p.add_argument("--output", dest="out", help=argparse.SUPPRESS)
        args = p.parse_args(argv[1:])
        try:
            export_gfpgan_onnx(version=args.version, model_path=args.model_path, out_path=args.out)
        except NotImplementedError as e:
            print(str(e))
            print("\nTip: Install ORT extras for validation: pip install -e \".[ort]\"\n"
                  "Then try running gfpup doctor to check providers.")
            return 0
        return 0
    print(f"Unknown subcommand: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
