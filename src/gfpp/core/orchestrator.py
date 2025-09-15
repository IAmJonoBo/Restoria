from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gfpp.probe.quality import probe_quality


@dataclass
class Plan:
    backend: str
    params: Dict[str, Any]
    postproc: Dict[str, Any]
    reason: str
    quality: Dict[str, Optional[float]] = field(default_factory=dict)
    faces: Dict[str, Any] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)


DEFAULT_PLAN = Plan(backend="gfpgan", params={"weight": 0.5}, postproc={}, reason="default")


def plan(image_path: str, opts: Dict[str, Any]) -> Plan:
    """Lightweight, deterministic planner.

    - Probes NIQE/BRISQUE (best-effort)
    - Routes based on simple thresholds
    """
    # Defaults
    backend = str(opts.get("backend", "gfpgan"))
    params: Dict[str, Any] = {"weight": float(opts.get("weight", 0.5))}
    post: Dict[str, Any] = {}
    reason = "fallback"

    try:
        q = probe_quality(image_path)
    except Exception:
        q = None
    faces: Dict[str, Any] = {}
    # Optional face probe
    try:
        from gfpp.probe.faces import detect_faces  # type: ignore

        fstats = detect_faces(image_path)
        if isinstance(fstats, dict):
            faces = fstats
    except Exception:
        pass
    if q is None:
        return Plan(
            backend=backend,
            params=params,
            postproc=post,
            reason="quality_probe_unavailable",
            quality={},
            faces=faces,
            detail={"note": "quality probe failed"},
        )

    niqe = q.get("niqe")
    bris = q.get("brisque")

    # Simple rules (tunable): lower NIQE/BRISQUE is better
    if niqe is None and bris is None:
        reason = "no_quality_signal"
    elif (niqe is not None and niqe < 7.5) or (bris is not None and bris < 35):
        backend = "gfpgan"
        reason = "few_artifacts"
        params["weight"] = min(max(float(opts.get("weight", 0.5)), 0.0), 1.0)
    elif (niqe is not None and niqe >= 12) or (bris is not None and bris >= 55):
        # very degraded → prefer CodeFormer with higher fidelity weight (>= 0.6)
        backend = "codeformer"
        reason = "heavy_degradation"
        w = float(opts.get("weight", 0.7))
        params["weight"] = min(max(max(w, 0.6), 0.0), 1.0)
    else:
        backend = "gfpgan"
        reason = "moderate_degradation"
        # For moderate degradation, standardize to 0.6 for determinism
        params["weight"] = 0.6

    # Background upsampling suggestion
    post["background"] = opts.get("background", "realesrgan")
    detail = {
        "routing_rules": {
            "few_artifacts": "niqe < 7.5 or brisque < 35",
            "heavy_degradation": "niqe >= 12 or brisque >= 55",
            "moderate_degradation": "otherwise",
        },
        "decision_inputs": {"niqe": niqe, "brisque": bris, "face_count": faces.get("face_count")},
    }
    return Plan(
        backend=backend,
        params=params,
        postproc=post,
        reason=reason,
        quality={"niqe": niqe, "brisque": bris},
        faces=faces,
        detail=detail,
    )
