<!-- markdownlint-disable MD012 -->

# Quality metrics

Measure restoration quality with optional metrics and learn how to
interpret the results.

This guide covers both the new Restoria CLI and the transitional gfpup CLI.
All metrics are optional and degrade gracefully when dependencies are
missing.

## Quick start

Compute metrics during a run and write a metrics file next to outputs:

```bash
# Restoria (recommended)
restoria run --input photos/ --output out/ --metrics full

# Compatibility helper (maps to the same metrics under the hood)
gfpup run --input photos/ --output out/ --metrics full

```

Outputs include:

- out/metrics.json — per-image metrics (and a planning block when using
  Restoria)
- out/manifest.json — run manifest with runtime metadata
- Optional: CSV/HTML reports (gfpup only): add `--csv-out metrics.csv`
  and/or `--html-report report.html`

## Metric presets

- off — no extra metrics (fastest)
- fast — ArcFace identity (if available) only
- full — ArcFace + LPIPS-Alex + DISTS, plus no-reference IQA when available

```bash
restoria run --input photo.jpg --output out/ --metrics fast
restoria run --input photo.jpg --output out/ --metrics full
```

Tip: Install extras for metrics: `pip install -e .[metrics]`. ArcFace is
optional; if weights/deps are missing, identity will be skipped.

## Available metrics

### ArcFace identity cosine

- What it measures: Facial identity preservation between input and restored image
- Range: 0.0 – 1.0 (higher is better)
- Typical good range: 0.8 – 0.95

### LPIPS (AlexNet)

- What it measures: Perceptual similarity
- Range: ~0.0 – 1.0 (lower is better)
- Typical good range: 0.1 – 0.3

### DISTS

- What it measures: Structure + texture similarity
- Range: ~0.0 – 1.0 (lower is better)
- Typical good range: 0.1 – 0.25

### No‑reference IQA (best‑effort)

- NIQE / BRISQUE or other proxies, if the optional packages are present
- Recorded when available; not required for any workflow

## metrics.json structure

Restoria writes a lightweight metrics file:

```json
{
  "metrics": [
    {
      "input": "photos/a.jpg",
      "restored_img": "out/a.png",
      "metrics": {
        "arcface_cosine": 0.91,
        "lpips_alex": 0.18,
        "dists": 0.14,
        "niqe": 3.2,
        "runtime_sec": 1.87,
        "vram_mb": 2950
      }
    }
  ],
  "plan": {
    "backend": "gfpgan",
    "reason": "default_backend",
    "params": {"version": "1.4"}
  }
}
```

Notes:

- The top-level `plan` block is present when using Restoria. gfpup writes
  only `{"metrics": [...]}`.
- Missing metrics appear as absent or `null` values; we never hard-fail a
  run if optional metrics can’t be computed.

## Comparing backends

Run multiple backends and compare their metrics:

```bash
restoria run --input test/ --output out_gfpgan/ --backend gfpgan --metrics full
restoria run --input test/ --output out_codeformer/ \
  --backend codeformer --metrics full
restoria run --input test/ --output out_rfpp/ \
  --backend restoreformerpp --metrics full
```

Then aggregate the numbers in your own notebook/script. Example skeleton:

```python
import json
from pathlib import Path

def avg(vals):
    vs = [v for v in vals if isinstance(v, (int, float))]
    return sum(vs) / len(vs) if vs else None

def summarize(metrics_path):
    data = json.loads(Path(metrics_path).read_text())
    arc = []
    lp = []
    ds = []
    t = []
    for rec in data.get("metrics", []):
        m = rec.get("metrics", {})
        arc.append(m.get("arcface_cosine"))
        lp.append(m.get("lpips_alex"))
        ds.append(m.get("dists"))
        t.append(m.get("runtime_sec"))
    return {
        "avg_arcface": avg(arc),
        "avg_lpips": avg(lp),
        "avg_dists": avg(ds),
        "avg_time": avg(t),
    }

print("GFPGAN:", summarize("out_gfpgan/metrics.json"))
print("CodeFormer:", summarize("out_codeformer/metrics.json"))
print("RFPP:", summarize("out_rfpp/metrics.json"))
```

## Quality thresholds (examples)

- Professional: LPIPS < 0.2, DISTS < 0.15, ArcFace > 0.85
- Social/Personal: LPIPS < 0.3, DISTS < 0.25, ArcFace > 0.75
- Archive/Historical: LPIPS < 0.4, DISTS < 0.35, ArcFace > 0.65

Interpret thresholds as guidelines; always inspect visuals.

## Troubleshooting

!!! warning "Metrics computation failed"
    - Ensure extras are installed: `pip install -e .[metrics]`
    - Some metrics require a GPU; otherwise they may be slow or skipped
    - Try `--metrics fast` to compute identity only

!!! tip "Performance"
    - Metrics add overhead: use `--metrics off` for production speed
    - Consider computing metrics on a subset for QA

---

Next:

- Choose the right backend → (choose-backend.md)
- Optimize hardware performance → (../HARDWARE_GUIDE.md)
- Batch processing → (batch-processing.md)

