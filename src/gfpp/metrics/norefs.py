from __future__ import annotations

from typing import Optional


def brisque(path: str) -> Optional[float]:
    try:
        from imquality import brisque as _brisque  # type: ignore
        from PIL import Image  # type: ignore

        return float(_brisque.score(Image.open(path)))
    except Exception:
        pass
    try:
        import pybrisque  # type: ignore
        import cv2  # type: ignore

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return float(pybrisque.BRISQUE().score(img))
    except Exception:
        return None


def niqe(path: str) -> Optional[float]:
    try:
        import torch  # type: ignore
        import torchvision.transforms.functional as F  # type: ignore
        from PIL import Image  # type: ignore
        from piq import niqe as _niqe  # type: ignore

        im = Image.open(path).convert("RGB")
        t = F.to_tensor(im).unsqueeze(0)
        with torch.no_grad():
            return float(_niqe(t).item())
    except Exception:
        pass
    try:
        from skimage import img_as_float  # type: ignore
        from skimage.io import imread  # type: ignore
        from skimage.metrics import niqe as _sk_niqe  # type: ignore

        img = imread(path)
        return float(_sk_niqe(img_as_float(img)))
    except Exception:
        return None
