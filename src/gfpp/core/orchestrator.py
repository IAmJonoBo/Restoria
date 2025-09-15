from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from gfpp.probe.quality import probe_quality


@dataclass
class Plan:
    backend: str
    params: Dict[str, Any]
    postproc: Dict[str, Any]
    reason: str


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
    if q is None:
        return Plan(backend=backend, params=params, postproc=post, reason="quality_probe_unavailable")

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
        # very degraded
        backend = "codeformer"
        reason = "heavy_degradation"
        params["weight"] = min(max(float(opts.get("weight", 0.7)), 0.0), 1.0)
    else:
        backend = "gfpgan"
        reason = "moderate_degradation"
        params["weight"] = float(opts.get("weight", 0.6))

    # Background upsampling suggestion
    post["background"] = opts.get("background", "realesrgan")
    return Plan(backend=backend, params=params, postproc=post, reason=reason)
