from __future__ import annotations

import glob
import os
from typing import List


def list_inputs(spec: str) -> List[str]:
    """Accept file|dir|glob and return sorted list of image-like paths."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if os.path.isfile(spec):
        return [spec]
    if os.path.isdir(spec):
        return sorted(p for p in glob.glob(os.path.join(spec, "*")) if os.path.splitext(p)[1].lower() in exts)
    # glob pattern
    return sorted(p for p in glob.glob(spec) if os.path.splitext(p)[1].lower() in exts)


def load_image_bgr(path: str):
    try:
        import cv2  # type: ignore

        return cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_image(path: str, img, *, jpg_quality: int = 95, png_compress: int = 3, webp_quality: int = 90) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import cv2  # type: ignore

        ext = os.path.splitext(path)[1].lower()
        params = []
        if ext in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
        elif ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compress)]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, int(webp_quality)]
        return bool(cv2.imwrite(path, img, params))
    except Exception:
        return False
