from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

try:  # allow execution as module or script
    from .synthetic import ensure_synthetic_dataset
except ImportError:  # pragma: no cover - fallback when executed as script
    from synthetic import ensure_synthetic_dataset  # type: ignore

try:
    from restoria.cli.run import run_cmd
except Exception:  # pragma: no cover - fallback when running outside editable install
    from src.restoria.cli.run import run_cmd  # type: ignore


@dataclass
class BenchmarkJob:
    name: str
    input: str
    backend: str = "gfpgan"
    metrics: str = "fast"
    auto: bool = False
    compile: bool = False
    ort_providers: List[str] = field(default_factory=list)
    extra_args: List[str] = field(default_factory=list)


def load_jobs_from_config(path: Path) -> tuple[Path, List[BenchmarkJob]]:
    data = json.loads(path.read_text())
    output_dir = Path(data.get("output_dir", "bench/out"))
    jobs_raw = data.get("jobs", [])
    jobs: List[BenchmarkJob] = []
    for item in jobs_raw:
        jobs.append(
            BenchmarkJob(
                name=item["name"],
                input=item["input"],
                backend=item.get("backend", "gfpgan"),
                metrics=item.get("metrics", "fast"),
                auto=bool(item.get("auto", False)),
                compile=bool(item.get("compile", False)),
                ort_providers=list(item.get("ort_providers", []) or []),
                extra_args=list(item.get("extra_args", []) or []),
            )
        )
    return output_dir, jobs


def percentile(values: Sequence[float], pct: float) -> float | None:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    vals.sort()
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def summarise_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"image_count": len(records)}
    metrics_bucket: Dict[str, List[float]] = {}
    backend_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    for rec in records:
        backend = rec.get("backend")
        if backend:
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        plan = rec.get("plan") or {}
        reason = plan.get("reason") if isinstance(plan, dict) else None
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        metrics = rec.get("metrics", {}) or {}
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                metrics_bucket.setdefault(key, []).append(float(val))
    metric_summary: Dict[str, Dict[str, float]] = {}
    for key, values in metrics_bucket.items():
        metric_summary[key] = {
            "mean": mean(values),
            "p95": percentile(values, 95.0),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    summary["metrics"] = metric_summary
    summary["backend_counts"] = backend_counts
    if reason_counts:
        summary["plan_reason_counts"] = reason_counts
    return summary


def run_job(job: BenchmarkJob, output_root: Path) -> Dict[str, Any]:
    job_out = output_root / job.name
    job_out.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(job.input)
    if dataset_path.is_dir():
        ensure_synthetic_dataset(dataset_path)
    else:
        dataset_path.mkdir(parents=True, exist_ok=True)
        ensure_synthetic_dataset(dataset_path)

    args = [
        "--input",
        str(dataset_path / "*" if dataset_path.is_dir() else dataset_path),
        "--output",
        str(job_out),
        "--backend",
        job.backend,
        "--metrics",
        job.metrics,
    ]
    if job.auto:
        args.append("--auto")
    if job.compile:
        args.append("--compile")
    if job.ort_providers:
        args.extend(["--ort-providers", *job.ort_providers])
    if job.extra_args:
        args.extend(job.extra_args)

    prev_profile = os.environ.get("RESTORIA_PROFILE")
    if not prev_profile:
        os.environ["RESTORIA_PROFILE"] = "1"
    try:
        run_cmd(args)
    finally:
        if prev_profile is None:
            os.environ.pop("RESTORIA_PROFILE", None)
        else:
            os.environ["RESTORIA_PROFILE"] = prev_profile

    metrics_path = job_out / "metrics.json"
    if not metrics_path.exists():
        return {
            "name": job.name,
            "output_dir": str(job_out),
            "image_count": 0,
            "metrics": {},
        }
    payload = json.loads(metrics_path.read_text())
    records = payload.get("metrics", [])
    summary = summarise_records(records)
    summary.update(
        {
            "name": job.name,
            "backend": job.backend,
            "metrics_level": job.metrics,
            "auto": job.auto,
            "compile": job.compile,
            "ort_providers": job.ort_providers,
            "output_dir": str(job_out),
        }
    )
    return summary


def write_summary(output_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    summary = {
        "schema_version": "2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "jobs": list(summaries),
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    csv_path = output_dir / "benchmark_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "job",
            "image_count",
            "runtime_sec_mean",
            "runtime_sec_p95",
            "memory_peak_mb_mean",
            "memory_peak_mb_p95",
        ])
        for job in summaries:
            metrics = job.get("metrics", {})
            runtime = metrics.get("runtime_sec", {})
            memory = metrics.get("memory_peak_mb", {})
            writer.writerow(
                [
                    job.get("name"),
                    job.get("image_count", 0),
                    runtime.get("mean"),
                    runtime.get("p95"),
                    memory.get("mean"),
                    memory.get("p95"),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Restoria benchmarks")
    parser.add_argument("--config", type=Path, help="Path to benchmark config JSON")
    parser.add_argument("--input", type=str, help="Input folder or glob pattern")
    parser.add_argument("--backend", type=str, default="gfpgan")
    parser.add_argument("--output", type=str, default="bench/out")
    parser.add_argument("--metrics", type=str, default="fast")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--ort-providers",
        nargs="+",
        default=[],
        help="Preferred ONNX Runtime providers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        output_dir, jobs = load_jobs_from_config(args.config)
    else:
        if not args.input:
            args.input = "bench/datasets/stress"
        output_dir = Path(args.output)
        jobs = [
            BenchmarkJob(
                name="default",
                input=args.input,
                backend=args.backend,
                metrics=args.metrics,
                auto=args.auto,
                compile=args.compile,
                ort_providers=list(args.ort_providers or []),
            )
        ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for job in jobs:
        summaries.append(run_job(job, output_dir))

    write_summary(output_dir, summaries)
    print(f"Benchmark summary written to {output_dir / 'benchmark_summary.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()
