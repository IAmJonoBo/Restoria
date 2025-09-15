from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EngineDecision:
    engine: str
    rationale: Dict[str, Optional[float]]
    rule: str


def _estimate_face_count(path: str) -> Optional[int]:
    try:
        import cv2  # type: ignore

        img = cv2.imread(path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use built-in Haar cascade as a light-weight heuristic
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)
        return int(len(faces))
    except Exception:
        return None


def select_engine_for_image(path: str) -> EngineDecision:
    """Rule-based engine selector using light heuristics.

    - Measures Laplacian variance (blur proxy), optional BRISQUE/NIQE
    - Estimates face count via Haar cascade
    - Chooses among: gfpgan (default), codeformer (robust), restoreformer (identity-faithful)
    """
    from gfpgan.metrics.quality import quality_signals_from_path

    q = quality_signals_from_path(path)
    n_faces = _estimate_face_count(path)

    lapvar = q.get("lapvar") or 0.0
    brisque = q.get("brisque")
    niqe = q.get("niqe")

    # Normalize into a rough quality score (lower is worse) when available
    # Defaults/thresholds derived from common ranges; kept conservative.
    severe_blur = lapvar < 20.0  # very blurry
    moderate_blur = lapvar < 60.0
    very_poor_brisque = brisque is not None and brisque > 60
    poor_brisque = brisque is not None and brisque > 40
    very_poor_niqe = niqe is not None and niqe > 8.0
    poor_niqe = niqe is not None and niqe > 6.0
    many_faces = (n_faces or 0) >= 3

    # Rules (ordered):
    # 1) Severe degradation -> CodeFormer
    if severe_blur or very_poor_brisque or very_poor_niqe:
        return EngineDecision(
            engine="codeformer",
            rationale={"lapvar": q.get("lapvar"), "brisque": brisque, "niqe": niqe, "faces": float(n_faces or 0)},
            rule="severe_degradation",
        )

    # 2) Multiple faces and moderate degradation -> GFPGAN (good default)
    if many_faces and (moderate_blur or poor_brisque or poor_niqe):
        return EngineDecision(
            engine="gfpgan",
            rationale={"lapvar": q.get("lapvar"), "brisque": brisque, "niqe": niqe, "faces": float(n_faces or 0)},
            rule="multi_face_moderate",
        )

    # 3) Mostly sharp, single/dual face -> RestoreFormer (identity-faithful)
    if (not moderate_blur) and (n_faces in {1, 2}):
        return EngineDecision(
            engine="restoreformer",
            rationale={"lapvar": q.get("lapvar"), "brisque": brisque, "niqe": niqe, "faces": float(n_faces or 0)},
            rule="sharp_few_faces",
        )

    # 4) Default -> GFPGAN
    return EngineDecision(
        engine="gfpgan",
        rationale={"lapvar": q.get("lapvar"), "brisque": brisque, "niqe": niqe, "faces": float(n_faces or 0)},
        rule="default",
    )


__all__ = ["EngineDecision", "select_engine_for_image"]
