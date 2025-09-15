# ---
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
#     format_version: '1.3'
#     jupytext_version: 1.16.4
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# Benchmark: run available backends on a tiny sample and emit a gallery + CSV.
#
# This paired notebook is minimal-by-default to keep CI fast. Heavy ops are opt-in.
# Requirements:
# - Install extras: `pip install -e ".[metrics,arcface,ort]"`
# - Optional backends via extras: codeformer, restoreformerpp
#
# Notes for future extensions:
# - add LPIPS/DISTS scoring when installed
# - add per-face identity tracking
# - export a small HTML gallery

# %%
from __future__ import annotations
import json
from pathlib import Path

# Light availability check using registry (best-effort)
try:
    from gfpp.core.registry import list_backends
except Exception:
    list_backends = None  # type: ignore

SAMPLES = [
    str(Path("samples/portrait.jpg")),
]
OUT_DIR = Path("bench_out")
OUT_DIR.mkdir(exist_ok=True)

record = {"samples": SAMPLES, "results": []}

if list_backends is not None:
    avail = list_backends(include_experimental=False)
else:
    avail = {"gfpgan": True}

# Choose a minimal set of backends for a quick pass
backends = [k for k, v in avail.items() if v]
if not backends:
    backends = ["gfpgan"]

# Dry-run path if heavy deps missing
print("Backends under test:", backends)

# Write a small manifest-like JSON for external tooling
with open(OUT_DIR / "benchmark_manifest.json", "w") as f:
    json.dump({"backends": backends, "notes": "Stub benchmark; fill in more steps."}, f, indent=2)

print("Benchmark stub prepared. See TODOs in this file to extend.")
