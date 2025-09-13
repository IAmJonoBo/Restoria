# Colab Guide

Open the notebook:
- https://colab.research.google.com/github/IAmJonoBo/GFPGAN/blob/main/notebooks/GFPGAN_Colab.ipynb

Features
- Install cell sets up Torch + Basicsr master (for torchvision compatibility)
- Interactive UI for:
  - Uploading images
  - Fetching images from URLs
  - Optional Drive mount
  - Selecting version, upscale, weight, and options
  - Running inference and previewing results
  - ZIP download of the results directory
- First-image before/after slider when original is available

Tips
- Use a GPU runtime to enable background upsampling (Real-ESRGAN) and for speed.
- The notebook prints versions of Torch, Torchvision, and Basicsr to help debugging.

## CI Smoke Mode

- The Colab notebook supports a lightweight smoke run for CI.
- Set the environment variable `NB_CI_SMOKE=1` to skip heavy installation, cloning, and inference cells.
- Example (local):
  - `NB_CI_SMOKE=1 pytest -c /dev/null --nbmake --nbmake-kernel=python3 --ignore=tests notebooks/GFPGAN_Colab.ipynb`
