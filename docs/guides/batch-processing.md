<!-- markdownlint-disable MD031 MD032 MD046 MD013 -->

# Batch processing

Process entire folders of images with consistent settings and quality tracking.

## Quick start

Restore all images in a folder:

```bash
restoria run --input photos/ --output results/ --backend gfpgan
```

GFPGAN automatically processes all supported image formats (JPG, PNG, WEBP) in the input folder.

## Basic batch operations

### Process a folder

=== "Default settings"
    ```bash
    restoria run --input photos/ --backend gfpgan
    ```

=== "Custom output folder"
    ```bash
    restoria run --input photos/ --output restored_photos/ --backend gfpgan
    ```

=== "With quality metrics"
    ```bash
    restoria run --input photos/ --metrics fast --output batch_fast/
    ```

### Supported file formats

GFPGAN processes these image formats:

- **JPEG/JPG** - Most common format
- **PNG** - Lossless format, good for high quality
- **WEBP** - Modern format with good compression
- **BMP** - Uncompressed bitmap
- **TIFF** - High quality, often used professionally

## Advanced batch processing

### Choose processing backend

=== "Quality-focused"
    ```bash
    # Best quality, slower processing
        restoria run \
            --input photos/ \
            --backend restoreformerpp \
            --metrics full \
            --output out_rfpp/
    ```

=== "Speed-focused"
    ```bash
    # Faster processing, good quality
    restoria run --input photos/ --backend codeformer --output out_cf/
    ```

=== "Balanced"
    ```bash
    # Good balance of speed and quality
    restoria run --input photos/ --backend gfpgan --metrics fast --output out_v14/
    ```

### Batch with background enhancement

=== "Full enhancement"
    ```bash
    # Restore faces (background handled by backend pipeline if available)
    restoria run --input photos/ --backend gfpgan --output out/
    ```

=== "Face-only (faster)"
    ```bash
    # Keep minimal processing for speed
    restoria run --input photos/ --backend codeformer --output out_fast/
    ```

### Device and memory management

=== "Auto device selection"
    ```bash
    restoria run --input photos/ --device auto --backend gfpgan
    ```

=== "Force CPU (low memory)"
    ```bash
    restoria run --input photos/ --device cpu --backend codeformer
    ```

=== "GPU with memory limits"
    ```bash
    # Prefer lighter backends if GPU memory is limited
    restoria run --input photos/ --backend codeformer --output out/
    ```

## Quality tracking and metrics

### Enable quality metrics

=== "Fast metrics (LPIPS only)"
    ```bash
    restoria run --input photos/ --metrics fast --output out_fast/
    ```

=== "Detailed metrics (LPIPS, DISTS, ArcFace)"
    ```bash
    restoria run --input photos/ --metrics full --output out_full/
    ```

=== "No metrics (fastest)"
    ```bash
    restoria run --input photos/ --metrics off --output out_nometrics/
    ```

### Understanding metric reports

The generated JSON report includes:

```json
{
    "summary": {
        "total_images": 150,
        "successful": 147,
        "failed": 3,
        "avg_lpips": 0.234,
        "avg_processing_time": 2.3
    },
    "per_image": {
        "photo001.jpg": {
            "lpips": 0.198,
            "dists": 0.156,
            "arcface_similarity": 0.892,
            "processing_time": 2.1,
            "faces_detected": 1
        }
    }
}
```

**Metric meanings:**

- **LPIPS**: Lower is better (perceptual similarity)
- **DISTS**: Lower is better (structural similarity)
- **ArcFace**: Higher is better (identity preservation)

## Provenance and reproducibility

### Deterministic processing

For reproducible results across runs:

```bash
restoria run \
    --input photos/ \
    --seed 42 \
    --deterministic \
    --backend gfpgan \
    --output out/
```

### Preserve metadata

Keep original EXIF data:

```bash
# Restoria preserves core provenance via manifest.json.
# EXIF preservation may depend on backend.
```

### Provenance tracking

GFPGAN automatically saves processing information:

```text
out/
├── restored_imgs/           # Processed images (if backend writes them separately)
├── metrics.json             # Per-image metrics (+ plan for Restoria)
└── manifest.json            # Run manifest (args, device, runtime env)
```

## Monitoring progress

### Progress indicators

=== "Simple progress bar"
    ```bash
    restoria run --input photos/ --backend gfpgan --output out/
    ```

=== "Detailed logging"
    ```bash
    restoria run --input photos/ --backend gfpgan --output out/
    ```

=== "Silent mode"
    ```bash
    restoria run --input photos/ --backend gfpgan --output out/
    ```

### Handling interruptions

If processing is interrupted:

```bash
# Resume strategy varies by backend; Restoria records args/manifests for reproducibility.
```

## Presets and automation

### Create processing presets

Save commonly used settings:

=== "High quality preset (example alias)"

```bash
# Create alias or script
alias restoria-hq='restoria run --backend restoreformerpp --metrics full'
```
```bash

```bash
# Use the preset
restoria-hq --input photos/ --output results/
```

=== "Fast processing preset (example alias)"

```bash
alias restoria-fast='restoria run --backend codeformer --metrics fast'
```
```bash

```bash
restoria-fast --input photos/ --output results/
```

### Automation scripts

=== "Bash script example"

```bash
#!/bin/bash
# batch_restore.sh

INPUT_DIR="$1"
OUTPUT_DIR="$2"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

restoria run \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --backend gfpgan \
    --metrics fast
```

=== "Python script example"

```python
#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def batch_restore(input_dir, output_dir, backend="gfpgan"):
    cmd = [
        "restoria", "run",
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--backend", backend,
        "--metrics", "fast",
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    batch_restore(sys.argv[1], sys.argv[2])
```

## Troubleshooting batch jobs

### Common issues

!!! error "Out of memory during batch processing"
    **Solutions:**

    ```bash
    # Use CPU mode
    restoria run --input photos/ --device cpu --backend codeformer
    ```

    ```bash
    # Prefer lighter backend
    restoria run --input photos/ --backend codeformer
    ```

    ```bash
    # Skip metrics
    restoria run --input photos/ --metrics off --backend gfpgan
    ```

!!! warning "Some images failed to process"
    Check the processing log:

    ```bash
    restoria run --input photos/ --backend gfpgan --output out/
    ```
    Common causes:
    - Corrupted image files
    - Unsupported format
    - No faces detected

!!! info "Processing is too slow"
    **Speed optimizations:**

    ```bash
    # Use faster backend
    gfpgan-infer --input photos/ --backend codeformer

    # Skip metrics
    gfpgan-infer --input photos/ --metrics none

    # Disable background processing
    gfpgan-infer --input photos/ --bg_upsampler none
    ```

### Performance tips

1. **Sort by file size**: Process smaller images first to get quick feedback
2. **Use SSD storage**: Faster I/O improves batch processing speed
3. **Monitor GPU memory**: Use `nvidia-smi` to watch memory usage
4. **Parallel processing**: For very large batches, split into chunks

---

**Next steps:**

- [Measure quality with metrics →](metrics.md)
- [Choose the right backend →](choose-backend.md)
- [Optimize hardware performance →](../HARDWARE_GUIDE.md)
