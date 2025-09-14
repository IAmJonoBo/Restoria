from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobSpec(BaseModel):
    input: str = Field(..., description="file|dir|glob; server-local path for now")
    backend: str = Field("gfpgan")
    background: str = Field("realesrgan")
    quality: str = Field("balanced", description="quick|balanced|best")
    preset: str = Field("natural")
    compile: str = Field("none")
    seed: Optional[int] = None
    deterministic: bool = False
    metrics: str = Field("off")
    output: str = Field("results")
    dry_run: bool = Field(False, description="If true, simulate run without loading heavy models")
    model_path_onnx: Optional[str] = Field(None, description="Path to ONNX model (for ORT backends)")
    auto_backend: bool = Field(False, description="Select backend per-image using heuristics")
    identity_lock: bool = Field(False, description="Retry with stricter preset if identity drops")
    identity_threshold: float = Field(0.25, description="Threshold for identity cosine to trigger retry")
    optimize: bool = Field(False, description="Try multiple weights and pick best by metric")
    weights_cand: str = Field("0.3,0.5,0.7", description="Comma-separated candidate weights for optimize")


class MetricCard(BaseModel):
    input: str
    restored_img: Optional[str]
    metrics: Dict[str, Any] = {}


class JobStatus(BaseModel):
    id: str
    status: str
    progress: float = 0.0
    result_count: int = 0
    results_path: Optional[str] = None
    error: Optional[str] = None


class Result(BaseModel):
    job: JobStatus
    metrics: List[MetricCard] = []
