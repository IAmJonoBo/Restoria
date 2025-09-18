# CLI Usage

Primary CLI: `restoria` (recommended)

Compatibility shims:

- `gfpup` — modular CLI powered by `gfpp`
- `gfpgan-infer` — legacy GFPGAN CLI kept for backward compatibility

## restoria run

Key flags:

- `--input` file or folder of images
- `--backend` one of: gfpgan, codeformer, restoreformerpp, ensemble, ...
- `--auto` enable planner to choose a backend and normalized params (off by default; without it the requested backend is preserved)
- `--metrics` off|fast|full (fast = ArcFace; full = ArcFace + LPIPS + DISTS)
- `--device` auto|cpu|cuda|mps
- `--dry-run` parse/plan and write manifest without executing
- `--plan-only` print the plan and exit (no IO)
- `--compile` try torch.compile (safe fallback)
- `--ort-providers` providers for ORT when applicable
- Precision and tiling (optional and safe-by-default):
  - `--precision {auto,fp16,bf16,fp32}`
  - `--tile <size>` and `--tile-overlap <pixels>`
  - Flags degrade gracefully when unsupported on your hardware
  - `--detector` can select between available face detectors (default: retinaface)

Examples (Restoria):

```bash
# Basic run with GFPGAN backend
restoria run --input inputs/whole_imgs --backend gfpgan --output out/

# Let the planner pick a backend and parameters
restoria run --input samples/portrait.jpg --auto --metrics fast --output out/

# Plan-only / dry-run
restoria run --input samples/portrait.jpg --plan-only
restoria run --input samples/portrait.jpg --dry-run --output out/

# Deterministic with seed
restoria run --input samples/portrait.jpg --seed 123 --deterministic
```

Outputs:

- `manifest.json` — args, runtime, results, models used
- `metrics.json` — per-image metrics (values may be None if unavailable)

Notes:

- Optional features degrade gracefully; missing metrics result in `null`
  values rather than errors.
- Heavy libraries (torch, cv2) are imported lazily to keep startup fast.
- CodeFormer is non-commercial (NTU S-Lab 1.0). It’s blocked by default and
  requires `--allow-noncommercial` or `RESTORIA_ALLOW_NONCOMMERCIAL=1`.
- External backends can be added via Python entry points under the group
  `gfpp.restorers` (mapping `name -> module:Class`). Discovered plugins are
  merged with built-ins; plugin errors are isolated.

## gfpup run (compatibility shim)

Examples:

```bash
gfpup run --input inputs/whole_imgs --backend gfpgan --output out/
gfpup run --input samples/portrait.jpg --auto --metrics fast --output out/
gfpup run --input samples/portrait.jpg --plan-only
gfpup run --input samples/portrait.jpg --dry-run --output out/
```

Notes:

- Mirrors Restoria behavior; heavy features degrade gracefully.
- Keep flags stable; may be removed in a future major.

## Legacy shim: gfpgan-infer

```bash
usage: gfpgan-infer [-i INPUT] [-o OUTPUT] [-v VERSION] [-s UPSCALE]
                    [--bg_upsampler BG_UPSAMPLER] [--bg_tile BG_TILE]
                    [--suffix SUFFIX] [--only_center_face] [--aligned]
                    [--ext EXT] [-w WEIGHT]
                    [--device {auto,cpu,cuda}] [--dry-run] [--no-download]
                    [--model-path MODEL_PATH] [--seed SEED] [--no-cmp]
```

Common examples:

- `gfpgan-infer -i inputs/whole_imgs -o results -v 1.4 -s 2 --device auto`
- `gfpgan-infer -i my.jpg -v 1.3 --no-download --model-path ./weights/GFPGANv1.3.pth`
- `gfpgan-infer --dry-run -v 1.4` (validate and exit)

Notes:

- On CPU, Real-ESRGAN background upsampling may be disabled for speed.
- `--seed` sets seeds for random, numpy, and torch.
- The shim prints a deprecation warning to stderr but maintains flag behavior
  and output layout for backward compatibility.
