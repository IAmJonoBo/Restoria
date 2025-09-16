Development Environment (uv)

Overview
- Uses uv for fast, reproducible Python environments and locking.
- Default target: Python 3.10 with a CPU-only stack compatible with BasicSR 1.4.2.

Prereqs
- Install uv: macOS `brew install uv`, Linux `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Python 3.10 recommended. 3.11+ is experimental (see below).

Quick Start (3.11, Torch 2.x default)
1) Sync deps (with dev tools):
   - `scripts/setup_uv.sh --python 3.11 --track torch2`
2) Run tests:
   - `scripts/test.sh`
   - Optional progress during tests:
     - Console: `scripts/test.sh --progress-console`
     - JSONL file: `scripts/test.sh --progress-log .pytest-progress.jsonl`
3) Lint / format:
   - `scripts/lint.sh`
   - `scripts/fmt.sh`

Notes
- The runtime depends on `torch` and `torchvision`. For compatibility with `basicsr==1.4.2`, we supply a Torch 1.x extra (`torch1`) and version markers to keep NumPy/Skimage/OpenCV in a compatible range for Python 3.10.
- If you install via pip directly, you can use the constraints file: `pip install -r requirements.txt -c constraints-3.10-compat.txt`.

Python 3.10 (Torch 1.x track; stable)
- For maximum compatibility with `basicsr==1.4.2` and CPU-only usage:
  - `scripts/setup_uv.sh --python 3.10 --track torch1`
  - Or with pip + constraints: `pip install -r requirements.txt -c constraints-3.10-compat.txt`

Torch 2.x notes
- Torch 2.x (3.11+) may require Basicsr master:
  - `uv pip install --no-cache-dir --upgrade "git+https://github.com/xinntao/BasicSR@master"`
  - Alternatively use `constraints-3.11-experimental.txt` with pip.

Apple Silicon
- All constraints are chosen to have prebuilt wheels on macOS arm64 where possible. If you hit build issues, ensure you’re on Python 3.10 and use the 3.10 constraints.
