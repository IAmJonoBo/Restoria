# Troubleshooting

- Basicsr import error with `functional_tensor`
  - Symptom: `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`
  - Fix: Install Basicsr from master (works with modern torchvision)
    - Pip: `pip uninstall -y basicsr && pip install --no-cache-dir --force-reinstall "git+https://github.com/xinntao/BasicSR@master"`
    - Colab: already handled in the install cell.

- Colab is slow or runs out of memory
  - Use a GPU runtime (Runtime → Change runtime type → GPU)
  - Keep background upsampling disabled on CPU (default in this fork)
  - Reduce `--upscale` or batch size (fewer/lighter images)

- Heavy installs on CI/Colab
  - Prefer pinned Python (3.11) with Torch 2.x track
  - The tests (light) job avoids heavyweight tests and validates CLI and imports

- Deterministic results
  - Use `--seed` for reproducibility (random, numpy, torch)
  - Consider CPU for stricter determinism (or set CUDA deterministic options)

- Missing model weights
  - Use `--model-path` to point to local weights
  - Or remove `--no-download` to allow fetching from the release URLs

- NIQE/BRISQUE not showing up
  - These are optional no‑reference metrics; if their dependencies aren’t installed, the run will continue and values will be omitted or null.
  - To enable: install the `metrics` extra or the specific libraries used by your environment.
  - Dry‑run computes NIQE/BRISQUE too when `--metrics` is enabled, which helps validate setup without model weights.

- OpenCV import errors
  - The CLI and tests fall back to PIL for image I/O when OpenCV isn’t available.
  - If you need strict OpenCV behavior, install `opencv-python` or `opencv-python-headless` explicitly.
