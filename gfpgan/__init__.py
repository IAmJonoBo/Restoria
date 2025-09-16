"""GFPGAN top-level package.

This package intentionally keeps imports light to avoid pulling in heavy
dependencies (torch, cv2, basicsr) at import time. Submodules should
perform their own lazy imports within functions/methods.

Public modules are available under the package namespace via standard
"import gfpgan.<module>" usage without side effects.
"""

from .version import __version__  # noqa: F401

__all__ = ["__version__"]
