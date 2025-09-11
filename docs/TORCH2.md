Torch 2.x Migration Path (Experimental)

Goals
- Support Python 3.11+ and Torch/Torchvision 2.x while keeping the stable 3.10 + Torch 1.x path intact.

Status
- GFPGAN code imports from `torchvision.ops` and `torchvision.transforms.functional`, which are compatible with 0.17–0.19.
- Known blocker was in `basicsr==1.4.2` using `torchvision.transforms.functional_tensor` removed in torchvision 0.15+. Upstream master addresses this.

How to Try
1) Create a uv env for Python 3.11 with Torch 2.x extra:
   - `uv sync -p 3.11 -E dev -E torch2`
2) If you hit Basicsr import errors, install master:
   - `uv pip install --no-cache-dir --upgrade "git+https://github.com/xinntao/BasicSR@master"`
3) Run tests:
   - `uv run pytest -q`

Notes
- Some torchvision ops (e.g., `roi_align`) rely on compiled C++/CUDA ops. CPU wheels include C++ kernels; CUDA requires matching CUDA wheels.
- If you need CUDA, install the appropriate Torch/Torchvision CUDA wheels and re-sync the environment.
- If downstream regressions appear (numerics, tolerance), we can add conditional branches or loosen test tolerances for the Torch 2.x track.

Next Steps (if adopting Torch 2.x by default)
- Change the default scripts to `--track torch2` and set CI matrix to Python 3.11.
- Remove Torch 1.x constraints when Basicsr master (or a tagged release) is required and stable.

