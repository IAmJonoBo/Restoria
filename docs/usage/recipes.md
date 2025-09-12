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
