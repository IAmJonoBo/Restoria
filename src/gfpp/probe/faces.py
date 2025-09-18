from __future__ import annotations

from typing import Dict, List, Optional


def _detect_with_insightface(path: str) -> Optional[Dict[str, object]]:
    try:
        import cv2  # type: ignore
        from insightface.app import FaceAnalysis  # type: ignore

        img = cv2.imread(path)
        if img is None:
            return None
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # type: ignore[arg-type]
        try:
            app.prepare(ctx_id=0, det_size=(640, 640))  # type: ignore[attr-defined]
        except Exception:
            try:
                app.prepare(ctx_id=0)  # type: ignore[attr-defined]
            except Exception:
                app.prepare()  # type: ignore[attr-defined]
        faces = app.get(img)
        sizes: List[int] = []
        try:
            seq = faces or []
        except Exception:
            seq = []
        for f in seq:
            bb = getattr(f, "bbox", None)
            if bb is None and isinstance(f, (list, tuple)) and len(f) >= 4:
                bb = f[:4]
            if bb is None:
                continue
            _, y1, _, y2 = [int(v) for v in bb[:4]]
            sizes.append(int(max(0, y2 - y1)))
        return {"count": int(len(sizes)), "sizes": sizes, "occlusions": []}
    except Exception:
        return None


def _detect_with_facexlib(path: str) -> Optional[Dict[str, object]]:
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
        sizes2: List[int] = []
        for _, y1, __, y2, ___ in bboxes:
            sizes2.append(int(max(0, min(img.shape[0], y2) - max(0, y1))))
        return {"count": int(len(sizes2)), "sizes": sizes2, "occlusions": []}
    except Exception:
        return None


def detect_faces(path: str) -> Optional[Dict[str, object]]:
    """Return a dictionary with face count and sizes if detectors are available.

    Structure: {"count": int, "sizes": List[int], "occlusions": Optional[List[float]]}
    Returns None if no detector is available.

    Preference order:
      1) insightface SCRFD (if available)
      2) facexlib retinaface (fallback)
    """
    # Try SCRFD, then facexlib
    res = _detect_with_insightface(path)
    if isinstance(res, dict):
        return res
    res = _detect_with_facexlib(path)
    if isinstance(res, dict):
        return res
    return None
