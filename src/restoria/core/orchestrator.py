# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Dict

from .planner import DEFAULT_PLAN, Plan, compute_plan


def plan(image_path: str, opts: Dict[str, Any]) -> Plan:
    return compute_plan(image_path, opts)


__all__ = ["plan", "Plan", "DEFAULT_PLAN", "compute_plan"]
