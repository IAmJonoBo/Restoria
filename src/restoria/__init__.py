# SPDX-License-Identifier: Apache-2.0
"""
Restoria — intelligent image revival.

Lightweight top-level package; heavy imports are deferred to submodules.
"""

__all__ = ["__version__"]

try:
    # Reuse root VERSION file if present
    import os

    here = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    vf = os.path.join(here, "VERSION")
    __version__ = open(vf).read().strip()
except Exception:  # pragma: no cover
    __version__ = "0.1.0"
