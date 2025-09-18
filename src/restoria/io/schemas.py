from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PlanSummary(BaseModel):
    backend: str
    reason: str
    confidence: Optional[float] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)


class ResultRecord(BaseModel):
    input: str
    backend: Optional[str] = None
    restored_img: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    plan: Optional[PlanSummary] = None


SCHEMA_VERSION = "2"


class MetricsPayload(BaseModel):
    schema_version: str = SCHEMA_VERSION
    metrics: List[ResultRecord] = Field(default_factory=list)
    plan: Optional[PlanSummary] = None


class RunManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    args: Dict[str, Any]
    device: Optional[str] = None
    results: List[ResultRecord] = Field(default_factory=list)
    metrics_file: Optional[str] = None
    env: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "PlanSummary",
    "ResultRecord",
    "MetricsPayload",
    "RunManifest",
    "SCHEMA_VERSION",
    "plan_summary_from",
]


def plan_summary_from(plan_obj: Any) -> PlanSummary:
    params = dict(getattr(plan_obj, "params", {}) or {})
    quality = dict(getattr(plan_obj, "quality", {}) or {})
    return PlanSummary(
        backend=str(getattr(plan_obj, "backend", "gfpgan")),
        reason=str(getattr(plan_obj, "reason", "unknown")),
        confidence=getattr(plan_obj, "confidence", None),
        params=params,
        quality=quality,
    )
