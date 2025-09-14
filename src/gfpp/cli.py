from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from .background import build_realesrgan
from .io import RunManifest, list_inputs, load_image_bgr, save_image, write_manifest
from .metrics import ArcFaceIdentity, DISTSMetric, LPIPSMetric
from .restorers.gfpgan import GFPGANRestorer
from .restorers.codeformer import CodeFormerRestorer
from .restorers.restoreformerpp import RestoreFormerPP


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
    p = argparse.ArgumentParser(prog="gfpup run")
    p.add_argument("--input", required=True)
    p.add_argument("--backend", default="gfpgan", choices=["gfpgan", "gfpgan-ort", "codeformer", "restoreformerpp", "diffbir", "hypir"])
    p.add_argument("--background", default="realesrgan", choices=["realesrgan", "swinir", "none"])
    p.add_argument("--preset", default="natural", choices=["natural", "detail", "document"])
    p.add_argument("--compile", default="none", choices=["none", "default", "max"])
    p.add_argument("--engine-params", default=None, help="JSON string for engine params")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--metrics", default="off", choices=["off", "fast", "full"])
    p.add_argument("--csv-out", default=None, help="Write metrics CSV to this path")
    p.add_argument("--html-report", default=None, help="Write HTML report to this path")
    p.add_argument("--auto-backend", action="store_true", help="Select backend per-image using quality heuristics")
    p.add_argument("--dry-run", action="store_true", help="Simulate run without loading models (copy inputs to outputs)")
    p.add_argument("--quality", default="balanced", choices=["quick", "balanced", "best"], help="Quality vs speed preset")
    p.add_argument("--identity-lock", action="store_true", help="Retry with stricter preset if identity drops")
    p.add_argument("--identity-threshold", type=float, default=0.25)
    p.add_argument("--optimize", action="store_true", help="Try multiple weights and pick best by metric")
    p.add_argument("--weights-cand", default="0.3,0.5,0.7", help="Comma-separated candidate weights for --optimize")
    p.add_argument("--video", action="store_true")
    p.add_argument("--temporal", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--version", default="1.4")
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    _set_deterministic(args.seed, args.deterministic)

    # Background upsampler
    bg = None
    if args.background == "realesrgan":
        # Map quality preset to tile/precision
        bg = build_realesrgan(device=args.device)

    # Choose restorer (optionally auto per-file)
    if args.auto_backend:
        # pick initial; will refine per-image below
        chosen_backend = "gfpgan"
    else:
        chosen_backend = args.backend

    if chosen_backend == "gfpgan":
        rest = GFPGANRestorer(device=args.device, bg_upsampler=bg, compile_mode=args.compile)
    elif chosen_backend == "gfpgan-ort":
        from .restorers.gfpgan_ort import ORTGFPGANRestorer

        rest = ORTGFPGANRestorer(device=args.device, bg_upsampler=bg)
    elif chosen_backend == "codeformer":
        rest = CodeFormerRestorer(device=args.device, bg_upsampler=bg)
    elif chosen_backend == "restoreformerpp":
        rest = RestoreFormerPP(device=args.device, bg_upsampler=bg)
    else:
        print(f"[WARN] Backend {chosen_backend} not yet implemented; falling back to gfpgan")
        rest = GFPGANRestorer(device=args.device, bg_upsampler=bg, compile_mode=args.compile)
    # Presets (small, testable defaults)
    preset = args.preset
    preset_weight = {"natural": 0.5, "detail": 0.7, "document": 0.3}.get(preset, 0.5)

    cfg: Dict[str, Any] = {
        "version": args.version,
        "upscale": 2,
        "use_parse": True,
        "detector": "retinaface_resnet50",
        "weight": preset_weight,
        "no_download": args.no_download,
    }

    inputs = list_inputs(args.input)
    results = []

    # Dry-run path: copy inputs to outputs and optionally write metrics/report
    if args.dry_run:
        for pth in inputs:
            import time

            t0 = time.time()
            img = load_image_bgr(pth)
            if img is None:
                continue
            base, _ = os.path.splitext(os.path.basename(pth))
            out_img = os.path.join(args.output, f"{base}.png")
            save_image(out_img, img)
            rec = {"input": pth, "restored_img": out_img, "metrics": {"runtime_sec": time.time() - t0}}
            results.append(rec)

        # Write manifest + optional CSV/HTML
        metrics_file = os.path.join(args.output, "metrics.json") if args.metrics != "off" else None
        if metrics_file:
            with open(metrics_file, "w") as f:
                json.dump({"metrics": results}, f, indent=2)
        man = RunManifest(args=vars(args), device=args.device, results=results, metrics_file=metrics_file)
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
            print(f"[WARN] Failed to load: {pth}")
            continue
        cfg["input_path"] = pth
        import time
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
        if args.auto_backend:
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
        chosen_weight = cfg.get("weight")
        res = None
        if args.optimize:
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
                # Score: prefer ArcFace cosine (higher better), fallback to negative LPIPS (lower better -> higher score)
                score = None
                if args.metrics in {"fast", "full"} and arc and arc.available():
                    # identity vs input; use restored image path later but we operate in-memory for now
                    # Save temp files if needed for metric calculation via file paths
                    import tempfile
                    import cv2
                    td = tempfile.mkdtemp()
                    a = os.path.join(td, "in.png")
                    b = os.path.join(td, "out.png")
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
                    a = os.path.join(td, "in.png")
                    b = os.path.join(td, "out.png")
                    cv2.imwrite(a, img)
                    if r_try and r_try.restored_image is not None:
                        cv2.imwrite(b, r_try.restored_image)
                    else:
                        b = a
                    d = lpips.distance_from_paths(a, b)
                    score = -d if isinstance(d, float) else None
                # If no metrics, fallback to w proximity to preset
                if score is None:
                    score = -abs(w - preset_weight)
                if best_score is None or score > best_score:
                    best_score = score
                    best_res = r_try
                    best_w = w
            res = best_res
            chosen_weight = best_w
            cfg["weight"] = chosen_weight
        else:
            res = rest.restore(img, cfg)
        dur = time.time() - t0
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                vram_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))  # type: ignore
        except Exception:
            vram_mb = None

        base, ext = os.path.splitext(os.path.basename(pth))
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
        if chosen_weight is not None:
            rec["metrics"]["weight"] = chosen_weight
        # Identity lock: if identity below threshold and ArcFace available, retry with stricter preset
        if args.identity_lock and (args.metrics in {"fast", "full"}) and arc and arc.available():
            import tempfile
            import cv2
            td = tempfile.mkdtemp()
            a = os.path.join(td, "in.png")
            b = os.path.join(td, "out.png")
            cv2.imwrite(a, img)
            cv2.imwrite(b, res.restored_image if (res and res.restored_image is not None) else img)
            s0 = arc.cosine_from_paths(a, b)
            if s0 is not None and s0 < args.identity_threshold:
                # Try stricter: lower weight
                stricter = max(0.2, float(cfg.get("weight", preset_weight)) - 0.2)
                cfg_strict = dict(cfg)
                cfg_strict["weight"] = stricter
                r2 = rest.restore(img, cfg_strict)
                cv2.imwrite(b, r2.restored_image if (r2 and r2.restored_image is not None) else img)
                s1 = arc.cosine_from_paths(a, b)
                if s1 is not None and s1 > s0:
                    res = r2
                    save_image(out_img, res.restored_image if res.restored_image is not None else img)
                    rec["metrics"]["identity_retry"] = True
                    rec["metrics"]["weight"] = stricter
        # Compute metrics vs input/restored when possible
        if args.metrics != "off":
            if arc and arc.available():
                rec["metrics"]["arcface_cosine"] = arc.cosine_from_paths(pth, out_img)
            if lpips and lpips.available():
                rec["metrics"]["lpips_alex"] = lpips.distance_from_paths(pth, out_img)
            if dists and dists.available():
                rec["metrics"]["dists"] = dists.distance_from_paths(pth, out_img)
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


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: gfpup run --input <path> --output out/ [options]")
        return 2
    cmd = argv[0]
    if cmd == "run":
        return cmd_run(argv[1:])
    print(f"Unknown subcommand: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
