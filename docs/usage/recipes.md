# CLI Recipes

- CPU batch of images, disable background, cap to 20 images
  - `gfpgan-infer -i inputs/whole_imgs -o results -v 1.4 --device cpu --bg_upsampler none --max-images 20`

- GPU quality run with weight sweep and manifest
  - `gfpgan-infer -i my.jpg -o results -v 1.4 --device cuda --sweep-weight 0.3,0.5,0.7 --manifest results/manifest.json`

- Deterministic CUDA with seed (potentially slower)
  - `gfpgan-infer -i my.jpg -v 1.4 --device cuda --deterministic-cuda --seed 123`

- Change detector and parsing behavior
  - `gfpgan-infer -i img.jpg -v 1.3 --detector scrfd --no-parse`

- Print environment versions and run
  - `gfpgan-infer -i img.jpg -v 1.4 --print-env --verbose`

## Performance & Autopilot Tips

- Autopilot (`--auto`):
  - For common photos, try: `--auto --select-by sharpness`.
  - For portraits where identity matters: `--auto --select-by identity` (falls back to sharpness if identity backend is unavailable).
  - Autopilot tries a small set of model/weight combos (e.g., 1.2/1.3 with 0.3/0.5) and picks the best by the metric on the restored image.

- Hardware-aware defaults (`--auto-hw`):
  - On CUDA: sets `--bg_precision fp16` and tiles based on VRAM (0/600/400 for high/med/low VRAM).
  - On CPU: sets a sensible `--workers` value up to 4.

- CPU concurrency (`--workers N`):
  - Parallelizes images across processes; each worker maintains its own restorer.
  - Start with 2–4 workers depending on core count and memory; avoid high values.

- Quality/perf trade-offs:
  - Increase `--bg_tile` or disable background (`--bg_upsampler none`) if you encounter OOM.
  - Lower `--upscale` for faster throughput on large batches.
