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
