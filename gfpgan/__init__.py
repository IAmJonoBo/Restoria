"""GFPGAN top-level package.

This package intentionally keeps imports light to avoid pulling in heavy
dependencies (torch, cv2, basicsr) at import time. Submodules should
perform their own lazy imports within functions/methods.

Public modules are available under the package namespace via standard
"import gfpgan.<module>" usage without side effects.
"""

# Prefer generated version module if available; fall back to VERSION file.
try:  # pragma: no cover - trivial
    from .version import __version__  # type: ignore
except Exception:  # pragma: no cover - CI fallback path
    import os

    _ver = "0.0.0"
    try:
        # VERSION file lives at repo root next to this package directory
        _pkg_dir = os.path.dirname(__file__)
        _root = os.path.abspath(os.path.join(_pkg_dir, os.pardir))
        _version_file = os.path.join(_root, "VERSION")
        if os.path.exists(_version_file):
            with open(_version_file, "r", encoding="utf-8") as f:
                _ver = f.read().strip()
    except Exception:
        pass
    __version__ = _ver  # type: ignore

__all__ = ["__version__"]
