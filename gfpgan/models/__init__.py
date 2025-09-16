import importlib
from os import path as osp
import os

try:  # Prefer basicsr scandir if available
    from basicsr.utils import scandir as _scandir  # type: ignore
except Exception:  # lightweight fallback without basicsr
    def _scandir(dir_path):
        for name in os.listdir(dir_path):
            yield osp.join(dir_path, name)

# automatically scan and import model modules for registry
# scan all the files that end with '_model.py' under the model folder
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in _scandir(model_folder) if v.endswith("_model.py")]
# import all the model modules
_model_modules = []
for file_name in model_filenames:
    try:
        _model_modules.append(importlib.import_module(f"gfpgan.models.{file_name}"))
    except Exception:
        pass
