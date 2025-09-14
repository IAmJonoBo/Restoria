from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class BenchItem:
    input: str
    backend: str


def run(samples_dir: str, output_dir: str, backend: str = "gfpgan") -> List[dict]:
    try:
        from src.gfpp.cli import cmd_run  # type: ignore
    except Exception:
        from gfpp.cli import cmd_run  # type: ignore

    os.makedirs(output_dir, exist_ok=True)
    args = [
        "--input",
        os.path.join(samples_dir, "*"),
        "--backend",
        backend,
        "--metrics",
        "fast",
        "--output",
        output_dir,
    ]
    cmd_run(args)
    # Read metrics.json
    mpath = os.path.join(output_dir, "metrics.json")
    if os.path.isfile(mpath):
        with open(mpath, "r") as f:
            js = json.load(f)
        return js.get("metrics", [])
    return []


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.get("metrics", {}).keys()})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "restored_img", *keys])
        for r in rows:
            w.writerow([r.get("input"), r.get("restored_img"), *[r.get("metrics", {}).get(k) for k in keys]])


def main():  # pragma: no cover
    out_dir = "bench/out"
    os.makedirs(out_dir, exist_ok=True)
    rows = run("bench/samples", out_dir, backend="gfpgan")
    csv_path = os.path.join(out_dir, "bench.csv")
    write_csv(csv_path, rows)
    # Optional HTML report via Jinja2
    try:
        from jinja2 import Template  # type: ignore

        tpl_path = "bench/report.html.jinja"
        if os.path.isfile(tpl_path):
            with open(tpl_path, "r") as f:
                tpl = Template(f.read())
            keys = sorted({k for r in rows for k in r.get("metrics", {}).keys()})
            html = tpl.render(rows=rows, keys=keys)
            with open(os.path.join(out_dir, "report.html"), "w") as f:
                f.write(html)
    except Exception:
        pass
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
