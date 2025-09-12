from __future__ import annotations

import cv2
import numpy as np


def sharpness_score(img_bgr: np.ndarray) -> float:
    """Simple focus metric: variance of Laplacian on grayscale.

    Higher is sharper.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def identity_distance(_orig_bgr: np.ndarray, _rest_bgr: np.ndarray) -> float | None:  # pragma: no cover (optional)
    """Optional identity distance using InsightFace if available.

    Lower is better (distance).
    Returns None if backend is not available.
    """
    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception:
        return None
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(256, 256))
    o = app.get(_orig_bgr)
    r = app.get(_rest_bgr)
    if not o or not r:
        return None
    # use first face embeddings
    emb_o = o[0].normed_embedding
    emb_r = r[0].normed_embedding
    dist = float(np.linalg.norm(emb_o - emb_r))
    return dist
