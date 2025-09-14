from __future__ import annotations

from typing import Dict, List, Optional


def available_eps() -> List[str]:
    """Return available ONNX Runtime execution providers, or empty if ORT missing."""
    try:
        import onnxruntime as ort  # type: ignore

        return list(ort.get_available_providers())
    except Exception:
        return []


def select_best_ep() -> Optional[str]:
    eps = available_eps()
    # Rough priority list; adjust as we learn more
    for cand in [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "DmlExecutionProvider",
        "ROCMExecutionProvider",
        "CPUExecutionProvider",
    ]:
        if cand in eps:
            return cand
    return eps[0] if eps else None


def create_session(model_path: str, providers: Optional[List[str]] = None):
    """Create an ORT InferenceSession with chosen providers; returns None if ORT missing."""
    try:
        import onnxruntime as ort  # type: ignore

        if providers is None:
            best = select_best_ep()
            providers = [best] if best else None
        sess = ort.InferenceSession(model_path, providers=providers)
        return sess
    except Exception:
        return None


def session_info(sess) -> Dict[str, str]:
    try:
        return {
            "providers": ",".join(sess.get_providers()),
        }
    except Exception:
        return {"providers": "n/a"}
