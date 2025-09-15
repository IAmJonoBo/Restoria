"""Test configuration for GFPGAN.

Adds the local `src` directory to `sys.path` so that the modular `gfpp`
package (declared via `package-dir` in `pyproject.toml`) can be imported
when the project has not been installed in editable mode. This keeps CI
lightweight and avoids requiring an upfront `pip install -e .` just to
exercise pure-Python registry/orchestrator logic.

If the user installs the project (editable or wheel), this shim is
effectively a no-op because the distribution metadata will already
resolve `gfpp`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir():  # defensive; only modify path when src exists
    sys.path.insert(0, str(_SRC))
