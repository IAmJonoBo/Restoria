# Local Gradio App

Run a lightweight local UI for GFPGAN.

- Install (editable): `pip install -e .[dev]`
- Launch: `gfpgan-gradio --server-port 7860 --share`
  - Options: `--server-name 0.0.0.0` (default), `--share` for a public link.

Features
- Upload multiple images, pick version, device (auto/cpu/cuda), upscale, weight.
- Choose detector, enable/disable face parsing.
- Optional background upsampler with precision and tile controls (GPU).
- Displays restored images and device info.

Notes
- The app lazily loads models and uses the same inference logic as the CLI.
- For better results, download weights first: `gfpgan-download-weights -v 1.4`.

## Docker (local app)

Build and run the Gradio app via Docker (CPU):

- Build: `docker build -t gfpgan-app .`
- Run: `docker run --rm -p 7860:7860 gfpgan-app`

Then open http://localhost:7860.

Notes
- The image installs Torch 2.x CPU wheels and BasicSR master for compatibility.
- For GPU, consider a CUDA base image and matching Torch wheels (not included here).
