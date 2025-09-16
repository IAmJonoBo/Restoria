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
    confidence: float = 0.5
    quality: Dict[str, Optional[float]] = field(default_factory=dict)
    faces: Dict[str, Any] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)


DEFAULT_PLAN = Plan(backend="gfpgan", params={"weight": 0.5}, postproc={}, reason="default", confidence=0.5)


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
            confidence=0.2,
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

    # Face-aware adjustment: for many faces, bias toward GFPGAN for speed/stability
    face_count = None
    try:
        face_count = int(faces.get("face_count")) if isinstance(faces.get("face_count"), (int, float)) else None
    except Exception:
        face_count = None
    if reason == "heavy_degradation" and face_count is not None and face_count >= 3:
        backend = "gfpgan"
        reason = "heavy_degradation_many_faces"

    # Background upsampling suggestion
    post["background"] = opts.get("background", "realesrgan")
    # Optional experimental HYPIR routing rule (opt-in only)
    try:
        experimental = bool(opts.get("experimental", False))
    except Exception:
        experimental = False
    if experimental:
        try:
            from gfpp.core.registry import list_backends  # type: ignore

            avail = list_backends(include_experimental=True)
            hypir_ok = bool(avail.get("hypir"))
        except Exception:
            hypir_ok = False
        prompt = opts.get("prompt")
        # Rule A: prompt provided → prefer HYPIR
        if hypir_ok and isinstance(prompt, str) and len(prompt.strip()) > 0:
            backend = "hypir"
            reason = "experimental_hypir_prompt"
            params["prompt"] = prompt
            # Standardize texture richness when unspecified to keep determinism
            params.setdefault("texture_richness", 0.6)
            params.setdefault("identity_lock", False)
            # small confidence boost for explicit prompt intent (applied later)
            params.setdefault("_conf_boost", 0.05)
        # Rule B: moderate degradation, few faces → try HYPIR
        elif hypir_ok and reason == "moderate_degradation":
            fc = face_count if isinstance(face_count, int) else None
            if fc is None or fc <= 2:
                backend = "hypir"
                reason = "experimental_hypir_moderate"
                params.setdefault("texture_richness", 0.6)
                params.setdefault("identity_lock", False)
    # Confidence estimation (0..1), based on margin from thresholds and face info
    conf = 0.5
    if reason in {"few_artifacts", "moderate_degradation", "heavy_degradation", "heavy_degradation_many_faces"}:
        m_vals = []
        if reason.startswith("few_artifacts"):
            if isinstance(niqe, (int, float)):
                m_vals.append((7.5 - float(niqe)) / 5.0)
            if isinstance(bris, (int, float)):
                m_vals.append((35.0 - float(bris)) / 15.0)
        elif reason.startswith("heavy_degradation"):
            if isinstance(niqe, (int, float)):
                m_vals.append((float(niqe) - 12.0) / 5.0)
            if isinstance(bris, (int, float)):
                m_vals.append((float(bris) - 55.0) / 15.0)
        elif reason == "moderate_degradation":
            m_vals.append(0.0)
        margin = max(m_vals) if m_vals else 0.0
        conf = max(0.05, min(0.95, 0.5 + 0.25 * margin))
    # Adjust confidence with face signal presence
    if not faces:
        conf = max(0.05, conf - 0.05)
    elif isinstance(face_count, int) and face_count >= 1:
        conf = min(0.99, conf + 0.05)
    if reason == "heavy_degradation_many_faces":
        conf = max(0.05, conf - 0.05)

    detail = {
        "routing_rules": {
            "few_artifacts": "niqe < 7.5 or brisque < 35",
            "heavy_degradation": "niqe >= 12 or brisque >= 55",
            "heavy_degradation_many_faces": "heavy_degradation and face_count >= 3",
            "moderate_degradation": "otherwise",
            # experimental rules (active only with --experimental)
            "experimental_hypir_prompt": "experimental and prompt provided",
            "experimental_hypir_moderate": "experimental and moderate_degradation and face_count <= 2",
        },
        "decision_inputs": {"niqe": niqe, "brisque": bris, "face_count": faces.get("face_count")},
    }
    # Apply any requested confidence boost
    conf_boost = 0.0
    try:
        conf_boost = float(params.pop("_conf_boost", 0.0))
    except Exception:
        conf_boost = 0.0
    conf = min(0.99, float(conf) + max(0.0, conf_boost))
    return Plan(
        backend=backend,
        params=params,
        postproc=post,
        reason=reason,
        confidence=float(conf),
        quality={"niqe": niqe, "brisque": bris},
        faces=faces,
        detail=detail,
    )
