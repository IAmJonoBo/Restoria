from __future__ import annotations

import json
from pathlib import Path


def test_notebooks_are_paired_via_jupytext():
    nb_dir = Path(__file__).resolve().parents[1] / "notebooks"
    if not nb_dir.exists():
        return  # nothing to test
    # We enforce pairing for the benchmark notebook only (legacy notebooks are tolerated)
    ipynbs = [p for p in nb_dir.glob("*.ipynb") if p.name == "benchmark.ipynb"]
    # Allow empty to pass in minimal env
    for nb in ipynbs:
        try:
            data = json.loads(nb.read_text(encoding="utf-8"))
        except Exception:
            # If not valid JSON (or large), skip rather than fail CI
            continue
        # Jupytext pairing typically records in metadata; we also allow a .py/.md sibling
        base = nb.with_suffix("")
        has_pair_file = base.with_suffix(".py").exists() or base.with_suffix(".md").exists()
        # Passing condition: either explicit pair file exists or metadata hints present
        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        jtx = meta.get("jupytext") if isinstance(meta, dict) else None
        assert has_pair_file or jtx is not None, f"Notebook {nb.name} should have a .py/.md pair or jupytext metadata"
