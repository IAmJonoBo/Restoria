from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobSpec(BaseModel):
    input: str = Field(..., description="file|dir|glob; server-local path for now")
    backend: str = Field("gfpgan")
    background: str = Field("realesrgan")
    preset: str = Field("natural")
    compile: str = Field("none")
    seed: Optional[int] = None
    deterministic: bool = False
    metrics: str = Field("off")
    output: str = Field("results")
    dry_run: bool = Field(False, description="If true, simulate run without loading heavy models")


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
