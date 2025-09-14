from __future__ import annotations

import os
from typing import Dict, Optional


def _read_bgr(path: str):
    try:
        import cv2  # type: ignore

        return cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception:
        return None


def laplacian_variance_from_path(path: str) -> Optional[float]:
    """Simple sharpness proxy: variance of Laplacian. Higher = sharper.

    Returns None on failure.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        img = _read_bgr(path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
    except Exception:
        return None


def brisque_from_path(path: str) -> Optional[float]:
    """Compute BRISQUE score if available. Lower is better. None if unavailable.

    Tries common libraries: imquality.brisque, pybrisque.
    """
    try:
        # imquality (Pillow input)
        from imquality import brisque as _brisque  # type: ignore
        from PIL import Image  # type: ignore

        return float(_brisque.score(Image.open(path)))
    except Exception:
        pass
    try:
        # pybrisque (OpenCV input)
        import pybrisque  # type: ignore
        import cv2  # type: ignore

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        # pybrisque expects grayscale ndarray
        s = pybrisque.BRISQUE().score(img)
        return float(s)
    except Exception:
        return None


def niqe_from_path(path: str) -> Optional[float]:
    """Compute NIQE score if available. Lower is better. None if unavailable.

    Tries piq.niqe (torch) or skimage's implementation if present.
    """
    # Torch path via piq
    try:
        import torch  # type: ignore
        import torchvision.transforms.functional as F  # type: ignore
        from PIL import Image  # type: ignore
        from piq import niqe as _niqe  # type: ignore

        # piq expects torch tensor BCHW in [0,1]
        im = Image.open(path).convert("RGB")
        t = F.to_tensor(im).unsqueeze(0)
        with torch.no_grad():
            score = _niqe(t)
        return float(score.item())
    except Exception:
        pass

    # skimage 0.19+ has niqe in skimage.metrics
    try:
        from skimage import img_as_float  # type: ignore
        from skimage.io import imread  # type: ignore
        from skimage.metrics import niqe as _sk_niqe  # type: ignore

        img = imread(path)
        return float(_sk_niqe(img_as_float(img)))
    except Exception:
        return None


def quality_signals_from_path(path: str) -> Dict[str, Optional[float]]:
    """Compute a small set of quality signals, gracefully skipping if deps absent."""
    return {
        "lapvar": laplacian_variance_from_path(path),
        "brisque": brisque_from_path(path),
        "niqe": niqe_from_path(path),
        "filesize": float(os.path.getsize(path)) if os.path.isfile(path) else None,
    }


__all__ = [
    "laplacian_variance_from_path",
    "brisque_from_path",
    "niqe_from_path",
    "quality_signals_from_path",
]

