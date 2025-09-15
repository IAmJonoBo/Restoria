from __future__ import annotations

from typing import Dict, List, Optional


def detect_faces(path: str) -> Optional[Dict[str, object]]:
    """Return a dictionary with face count and sizes if detectors are available.

    Structure: {"count": int, "sizes": List[int], "occlusions": Optional[List[float]]}
    Returns None if no detector is available.
    """
    # Try facexlib first
    try:
        import cv2  # type: ignore
        from facexlib.detection import retinaface  # type: ignore

        img = cv2.imread(path)
        if img is None:
            return None
        detector = retinaface.RetinaFace()
        bboxes, _ = detector.detect(img, threshold=0.6)
        if bboxes is None:
            return {"count": 0, "sizes": [], "occlusions": []}
        sizes: List[int] = []
        for _, y1, __, y2, ___ in bboxes:
            sizes.append(int(max(0, min(img.shape[0], y2) - max(0, y1))))
        return {"count": int(len(sizes)), "sizes": sizes, "occlusions": []}
    except Exception:
        pass
    # Fallbacks could include other detectors; return None if unavailable
    return None
