# CLI Usage

```
usage: gfpgan-infer [-i INPUT] [-o OUTPUT] [-v VERSION] [-s UPSCALE]
                    [--bg_upsampler BG_UPSAMPLER] [--bg_tile BG_TILE]
                    [--suffix SUFFIX] [--only_center_face] [--aligned]
                    [--ext EXT] [-w WEIGHT]
                    [--device {auto,cpu,cuda}] [--dry-run] [--no-download]
                    [--model-path MODEL_PATH] [--seed SEED] [--no-cmp]
```

- Common examples
  - `gfpgan-infer -i inputs/whole_imgs -o results -v 1.4 -s 2 --device auto`
  - `gfpgan-infer -i my.jpg -v 1.3 --no-download --model-path ./weights/GFPGANv1.3.pth`
  - `gfpgan-infer --dry-run -v 1.4` (validate and exit)

- Notes
  - On CPU, Real-ESRGAN background upsampling is disabled for speed.
  - `--seed` sets seeds for random, numpy, and torch.
