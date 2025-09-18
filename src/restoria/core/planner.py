from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from gfpp.core.planner import DEFAULT_PLAN, Plan
from gfpp.core import planner as _gfpp_planner

base_compute_plan = _gfpp_planner.compute_plan


def compute_plan(image_path: str, opts: Dict[str, Any]) -> Plan:
    shared_plan = base_compute_plan(image_path, opts)
    params = dict(shared_plan.params)

    if bool(opts.get("compile", False)):
        params["compile"] = True

    ort_providers = opts.get("ort_providers")
    if isinstance(ort_providers, (list, tuple)) and ort_providers:
        params["ort_providers"] = list(ort_providers)

    auto_mode = bool(opts.get("auto", False))
    explicit_backend = opts.get("backend")

    backend = shared_plan.backend
    reason = shared_plan.reason

    if not auto_mode:
        if isinstance(explicit_backend, str) and explicit_backend:
            backend = explicit_backend
            reason = "user_selected"
        else:
            backend = shared_plan.backend
    else:
        if isinstance(explicit_backend, str) and explicit_backend and explicit_backend != "gfpgan":
            backend = explicit_backend

    return replace(shared_plan, backend=backend, params=params, reason=reason)


__all__ = ["compute_plan", "Plan", "DEFAULT_PLAN", "base_compute_plan"]
