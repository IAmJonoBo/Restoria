from __future__ import annotations

"""Shared planning logic for GFPP-based orchestrators."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gfpp.probe.quality import probe_quality

try:
    from gfpp.metrics.adv_quality import advanced_scores  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    advanced_scores = None  # type: ignore


@dataclass
class Plan:
    """Normalized execution plan returned by the planner."""

    backend: str
    params: Dict[str, Any]
    postproc: Dict[str, Any]
    reason: str
    confidence: float = 0.5
    quality: Dict[str, Optional[float]] = field(default_factory=dict)
    faces: Dict[str, Any] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)


DEFAULT_PLAN = Plan(backend="gfpgan", params={"weight": 0.5}, postproc={}, reason="default", confidence=0.5)


def _get_face_count(fd: Dict[str, Any]) -> int | None:
    try:
        if "count" in fd and isinstance(fd["count"], (int, float)):
            return int(fd["count"])
        if "face_count" in fd and isinstance(fd["face_count"], (int, float)):
            return int(fd["face_count"])
    except Exception:
        return None
    return None


def _median_face_size(faces: Dict[str, Any]) -> float | None:
    try:
        sizes = []
        if isinstance(faces.get("sizes"), list):
            for size in faces.get("sizes", []):
                try:
                    sizes.append(int(size))
                except Exception:
                    continue
        if not sizes:
            return None
        sizes.sort()
        mid = len(sizes) // 2
        if len(sizes) % 2 == 1:
            return float(sizes[mid])
        return float((sizes[mid - 1] + sizes[mid]) / 2.0)
    except Exception:
        return None


def _attach_confidence(base_conf: float, reason: str, niqe, brisque, faces: Dict[str, Any], params: Dict[str, Any]) -> float:
    conf = base_conf
    m_vals: list[float] = []
    if reason.startswith("few_artifacts"):
        if isinstance(niqe, (int, float)):
            m_vals.append((7.5 - float(niqe)) / 5.0)
        if isinstance(brisque, (int, float)):
            m_vals.append((35.0 - float(brisque)) / 15.0)
    elif reason.startswith("heavy_degradation"):
        if isinstance(niqe, (int, float)):
            m_vals.append((float(niqe) - 12.0) / 5.0)
        if isinstance(brisque, (int, float)):
            m_vals.append((float(brisque) - 55.0) / 15.0)
    elif reason == "moderate_degradation":
        m_vals.append(0.0)
    margin = max(m_vals) if m_vals else 0.0
    conf = max(0.05, min(0.95, 0.5 + 0.25 * margin))
    face_count = _get_face_count(faces) if isinstance(faces, dict) else None
    if not faces:
        conf = max(0.05, conf - 0.05)
    elif isinstance(face_count, int) and face_count >= 1:
        conf = min(0.99, conf + 0.05)
    if reason == "heavy_degradation_many_faces":
        conf = max(0.05, conf - 0.05)
    try:
        conf_boost = float(params.pop("_conf_boost", 0.0))
    except Exception:
        conf_boost = 0.0
    return min(0.99, float(conf) + max(0.0, conf_boost))


def compute_plan(image_path: str, opts: Dict[str, Any]) -> Plan:
    """Generate an execution plan with heuristic routing and annotations."""

    auto_mode = bool(opts.get("auto", False))
    backend = str(opts.get("backend", "gfpgan"))
    params: Dict[str, Any] = {"weight": float(opts.get("weight", 0.5))}
    post: Dict[str, Any] = {}
    reason = "fallback"

    try:
        quality = probe_quality(image_path)
    except Exception:
        quality = None

    adv: Dict[str, Optional[float]] = {}
    if advanced_scores is not None:
        try:
            raw_adv = advanced_scores(image_path)
            if isinstance(raw_adv, dict):
                for key, value in raw_adv.items():
                    try:
                        adv[key] = float(value) if isinstance(value, (int, float)) else None
                    except Exception:
                        adv[key] = None
        except Exception:
            adv = {}

    faces: Dict[str, Any] = {}
    try:
        from gfpp.probe.faces import detect_faces  # type: ignore

        face_stats = detect_faces(image_path)
        if isinstance(face_stats, dict):
            faces = face_stats
    except Exception:
        pass

    face_count = _get_face_count(faces) if isinstance(faces, dict) else None
    median_size = _median_face_size(faces)
    detector_choice = None
    if isinstance(face_count, int) and face_count >= 1:
        if face_count >= 3:
            detector_choice = "scrfd"
        elif median_size is not None and median_size < 80:
            detector_choice = "scrfd"
        else:
            detector_choice = "retinaface_resnet50"
        params["detector"] = detector_choice

    if quality is None:
        return Plan(
            backend=backend,
            params=params,
            postproc=post,
            reason="quality_probe_unavailable",
            confidence=0.2,
            quality={},
            faces=faces,
            detail={
                "note": "quality probe failed",
                "detector_decision": detector_choice,
                "decision_inputs": {"face_count": face_count, "face_median_size": median_size},
            },
        )

    niqe = quality.get("niqe")
    brisque = quality.get("brisque")

    if niqe is None and brisque is None:
        reason = "no_quality_signal"
    elif (niqe is not None and niqe < 7.5) or (brisque is not None and brisque < 35):
        backend = "gfpgan"
        reason = "few_artifacts"
        params["weight"] = min(max(float(opts.get("weight", 0.5)), 0.0), 1.0)
    elif (niqe is not None and niqe >= 12) or (brisque is not None and brisque >= 55):
        backend = "codeformer"
        reason = "heavy_degradation"
        weight = float(opts.get("weight", 0.7))
        params["weight"] = min(max(max(weight, 0.6), 0.0), 1.0)
    else:
        backend = "gfpgan"
        reason = "moderate_degradation"
        params["weight"] = 0.6

    if reason == "heavy_degradation" and isinstance(face_count, int) and face_count >= 3:
        backend = "gfpgan"
        reason = "heavy_degradation_many_faces"

    post["background"] = opts.get("background", "realesrgan")

    no_faces_detected = isinstance(face_count, int) and face_count == 0
    if no_faces_detected:
        try:
            from gfpp.core.registry import list_backends  # type: ignore

            available = list_backends(include_experimental=False)
            restoreformer_info = available.get("restoreformerpp") or {}
            restoreformer_available = bool(isinstance(restoreformer_info, dict) and restoreformer_info.get("available"))
        except Exception:
            restoreformer_available = False

        if restoreformer_available and auto_mode:
            backend = "restoreformerpp"
            reason = "no_faces_detected"
            params.setdefault("weight", 0.4)
            params.setdefault("_conf_boost", 0.05)

    experimental = bool(opts.get("experimental", False))
    if experimental:
        try:
            from gfpp.core.registry import list_backends  # type: ignore

            available = list_backends(include_experimental=True)
            hypir_ok = bool(available.get("hypir"))
        except Exception:
            hypir_ok = False
        prompt = opts.get("prompt")
        if hypir_ok and isinstance(prompt, str) and prompt.strip():
            backend = "hypir"
            reason = "experimental_hypir_prompt"
            params.setdefault("texture_richness", 0.6)
            params.setdefault("identity_lock", False)
            params["prompt"] = prompt
            params.setdefault("_conf_boost", 0.05)
        elif hypir_ok and reason == "moderate_degradation":
            if face_count is None or face_count <= 2:
                backend = "hypir"
                reason = "experimental_hypir_moderate"
                params.setdefault("texture_richness", 0.6)
                params.setdefault("identity_lock", False)

    confidence = _attach_confidence(0.5, reason, niqe, brisque, faces, params)
    quality_map: Dict[str, Optional[float]] = {"niqe": niqe, "brisque": brisque}
    try:
        if isinstance(adv, dict):
            for key in ("maniqa", "contrique"):
                if key in adv:
                    value = adv[key]
                    if isinstance(value, (int, float)):
                        quality_map[key] = float(value)
                    elif value is None:
                        quality_map[key] = None
    except Exception:
        pass

    detail = {
        "routing_rules": {
            "few_artifacts": "niqe < 7.5 or brisque < 35",
            "heavy_degradation": "niqe >= 12 or brisque >= 55",
            "heavy_degradation_many_faces": "heavy_degradation and face_count >= 3",
            "moderate_degradation": "otherwise",
            "experimental_hypir_prompt": "experimental and prompt provided",
            "experimental_hypir_moderate": "experimental and moderate_degradation with face_count <= 2",
            "no_faces_detected": "no faces detected and restoreformerpp available",
        },
        "decision_inputs": {
            "niqe": niqe,
            "brisque": brisque,
            "face_count": face_count,
            "face_median_size": median_size,
        },
        "detector_decision": detector_choice,
    }

    return Plan(
        backend=backend,
        params=params,
        postproc=post,
        reason=reason,
        confidence=confidence,
        quality=quality_map,
        faces=faces,
        detail=detail,
    )


# Backwards compatibility wrapper -------------------------------------------------

def plan(image_path: str, opts: Dict[str, Any]) -> Plan:
    """Alias maintained for legacy imports (gfpp.core.orchestrator.plan)."""

    return compute_plan(image_path, opts)
