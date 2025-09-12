# CLI Usage

```
usage: gfpgan-infer [-i INPUT] [-o OUTPUT] [-v VERSION] [-s UPSCALE]
                    [--bg_upsampler {realesrgan,none}] [--bg_tile BG_TILE]
                    [--bg_precision {auto,fp16,fp32}]
                    [--suffix SUFFIX] [--only_center_face] [--aligned]
                    [--ext EXT] [-w WEIGHT] [--sweep-weight SWEEP_WEIGHT]
                    [--jpg-quality JPG_QUALITY] [--png-compress PNG_COMPRESS]
                    [--webp-quality WEBP_QUALITY]
                    [--detector {retinaface_resnet50,retinaface_mobile0.25,scrfd}] [--no-parse]
                    [--device {auto,cpu,cuda}] [--dry-run] [--no-download]
                    [--model-path MODEL_PATH] [--seed SEED] [--no-cmp]
                    [--manifest MANIFEST] [--print-env]
                    [--deterministic-cuda] [--eye-dist-threshold EYE_DIST_THRESHOLD]
                    [--max-images MAX_IMAGES] [--skip-existing]
                    [--verbose]
```

- Common examples
  - `gfpgan-infer -i inputs/whole_imgs -o results -v 1.4 -s 2 --device auto`
  - `gfpgan-infer -i "inputs/whole_imgs/*.png" -v 1.3 --bg_upsampler none`
  - `gfpgan-infer -i my.jpg -v 1.3 --no-download --model-path ./gfpgan/weights/GFPGANv1.3.pth`
  - `gfpgan-infer --dry-run -v 1.4 --verbose` (validate and exit)
  - Weight sweep: `gfpgan-infer -i img.jpg -v 1.4 --sweep-weight 0.3,0.5,0.7 --manifest out.json`
  - Deterministic CUDA: `gfpgan-infer -i img.jpg -v 1.4 --deterministic-cuda --seed 123`
  - Quality controls: `--jpg-quality 95 --png-compress 3 --webp-quality 90`

- Notes
  - On CPU, Real-ESRGAN background upsampling is disabled automatically.
  - Use `--bg_precision fp32` to force full precision, or `fp16` to half on CUDA.
  - `--seed` sets seeds for random, numpy, and torch.
  - Use `--detector` to switch facexlib detectors; `--no-parse` disables face parsing.
  - `--manifest out.json` writes a manifest of inputs and outputs for automation.
  - `--skip-existing` avoids re-writing outputs; `--max-images` caps processing.
  - `--print-env` prints torch/torchvision/basicsr/facexlib versions and CUDA availability.
  - Quality controls apply by file extension (jpg/png/webp) when saving outputs.

Model weights
- Download utility: `gfpgan-download-weights --list` to view; `gfpgan-download-weights -v 1.4` to fetch.
- Destination: downloads to `gfpgan/weights/` by default.
