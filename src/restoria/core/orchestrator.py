# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class Plan:
    backend: str
    params: Dict[str, Any] = field(default_factory=dict)
    postproc: List[str] = field(default_factory=list)
    reason: str = "fixed"
    confidence: float | None = None


def plan(_image_path: str, opts: Dict[str, Any]) -> Plan:
    """Compute a simple execution plan.

    Recognized options in opts:
    - backend: str
    - experimental: bool
    - compile: bool (optional)
    - ort_providers: list[str] (optional)
    - prompt: str (optional)
    """
    backend = str(opts.get("backend", "gfpgan"))
    experimental = bool(opts.get("experimental", False))
    reason = "user_selected"
    if experimental and opts.get("prompt"):
        backend = "hypir"
        reason = "experimental_prompt_bias"

    params: Dict[str, Any] = {}
    # Thread-through optional performance params deterministically
    if bool(opts.get("compile", False)):
        params["compile"] = True
    ort_providers = opts.get("ort_providers")
    if isinstance(ort_providers, (list, tuple)) and ort_providers:
        params["ort_providers"] = list(ort_providers)

    return Plan(backend=backend, params=params, postproc=[], reason=reason, confidence=None)
