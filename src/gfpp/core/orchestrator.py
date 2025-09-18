from __future__ import annotations

"""Backwards-compatible orchestrator module.

All core planning logic lives in ``gfpp.core.planner``. This module exists to
preserve the legacy import surface (`Plan`, ``DEFAULT_PLAN``, ``plan``) and the
monkeypatch points used in older tests/plugins.
"""

from typing import Any, Dict

from gfpp.probe.quality import probe_quality  # re-exported for compatibility

try:
    from gfpp.metrics.adv_quality import advanced_scores  # type: ignore # pragma: no cover
except Exception:  # pragma: no cover - optional dependency
    advanced_scores = None  # type: ignore

from . import planner as _planner
from .planner import DEFAULT_PLAN, Plan


def plan(image_path: str, opts: Dict[str, Any]) -> Plan:
    """Delegate to the shared planner implementation while honouring monkeypatches."""

    # Sync patched globals into the shared planner so legacy monkeypatch hooks still work.
    if getattr(_planner, "probe_quality", None) is not probe_quality:
        _planner.probe_quality = probe_quality  # type: ignore[attr-defined]
    if getattr(_planner, "advanced_scores", None) is not advanced_scores:
        _planner.advanced_scores = advanced_scores  # type: ignore[attr-defined]
    return _planner.compute_plan(image_path, opts)


__all__ = ["Plan", "DEFAULT_PLAN", "plan", "probe_quality", "advanced_scores"]
