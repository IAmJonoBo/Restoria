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
    p.add_argument("--backend", default="gfpgan", choices=["gfpgan", "codeformer", "restoreformerpp", "diffbir", "hypir"])
    p.add_argument("--background", default="realesrgan", choices=["realesrgan", "swinir", "none"])
    p.add_argument("--preset", default="natural", choices=["natural", "detail", "document"])
    p.add_argument("--compile", default="none", choices=["none", "default", "max"])
    p.add_argument("--engine-params", default=None, help="JSON string for engine params")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--metrics", default="off", choices=["off", "fast", "full"])
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
        bg = build_realesrgan(device=args.device)

    # Choose restorer
    if args.backend == "gfpgan":
        rest = GFPGANRestorer(device=args.device, bg_upsampler=bg, compile_mode=args.compile)
    elif args.backend == "codeformer":
        rest = CodeFormerRestorer(device=args.device, bg_upsampler=bg)
    elif args.backend == "restoreformerpp":
        rest = RestoreFormerPP(device=args.device, bg_upsampler=bg)
    else:
        print(f"[WARN] Backend {args.backend} not yet implemented; falling back to gfpgan")
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
