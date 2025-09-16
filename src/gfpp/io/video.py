"""Minimal video IO helpers with graceful fallbacks.

Public API is intentionally small and stable; functions return None or
raise ValueError on bad inputs rather than importing heavy deps at import time.

No new hard dependencies are introduced. If cv2 is unavailable, readers/writers
will return None and callers should degrade gracefully.
"""
from __future__ import annotations
from typing import Optional, Tuple


def read_video_info(path: str) -> Optional[Tuple[int, int, float]]:
    """Return (width, height, fps) or None if backend unavailable or file invalid.

    This is a hook point only; callers must handle None and avoid raising.
    """
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return (w, h, fps)
    except Exception:
        return None


def ensure_writer(path: str, size: Tuple[int, int], fps: float):
    """Return a cv2.VideoWriter if available and opened; else None.

    size is (width, height). Caller owns lifecycle and must call release().
    """
    try:
        import cv2  # type: ignore

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, max(fps, 1.0), size)
        if writer is None or not writer.isOpened():
            return None
        return writer
    except Exception:
        return None
