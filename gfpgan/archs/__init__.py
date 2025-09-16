import importlib
from os import path as osp
import os

try:  # Prefer basicsr scandir if available
    from basicsr.utils import scandir as _scandir  # type: ignore
except Exception:  # lightweight fallback without basicsr
    def _scandir(dir_path):
        for name in os.listdir(dir_path):
            yield osp.join(dir_path, name)

# automatically scan and import arch modules for registry
# scan all the files that end with '_arch.py' under the archs folder
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in _scandir(arch_folder) if v.endswith("_arch.py")]
# import all the arch modules (best-effort, keep lightweight)
_arch_modules = []
for file_name in arch_filenames:
    try:
        _arch_modules.append(importlib.import_module(f"gfpgan.archs.{file_name}"))
    except Exception:
        # Avoid failing import-time if optional deps missing
        pass
