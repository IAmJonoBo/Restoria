# Batch processing

Process entire folders of images with consistent settings and quality tracking.

## Quick start

Restore all images in a folder:

```bash
gfpgan-infer --input photos/ --output results/ --version 1.4
```

GFPGAN automatically processes all supported image formats (JPG, PNG, WEBP) in the input folder.

## Basic batch operations

### Process a folder

=== "Default settings"
    ```bash
    gfpgan-infer --input photos/ --version 1.4
    ```

=== "Custom output folder"
    ```bash
    gfpgan-infer --input photos/ --output restored_photos/ --version 1.4
    ```

=== "With quality metrics"
    ```bash
    gfpgan-infer --input photos/ --metrics fast --report-path batch_report.json
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
    gfpgan-infer --input photos/ --backend restoreformer --metrics detailed
    ```

=== "Speed-focused"
    ```bash
    # Faster processing, good quality
    gfpgan-infer --input photos/ --backend codeformer --bg_upsampler none
    ```

=== "Balanced"
    ```bash
    # Good balance of speed and quality
    gfpgan-infer --input photos/ --version 1.4 --metrics fast
    ```

### Batch with background enhancement

=== "Full enhancement"
    ```bash
    # Restore faces and enhance backgrounds
    gfpgan-infer --input photos/ --bg_upsampler realesrgan --upscale 2
    ```

=== "Face-only (faster)"
    ```bash
    # Skip background processing for speed
    gfpgan-infer --input photos/ --bg_upsampler none
    ```

### Device and memory management

=== "Auto device selection"
    ```bash
    gfpgan-infer --input photos/ --device auto
    ```

=== "Force CPU (low memory)"
    ```bash
    gfpgan-infer --input photos/ --device cpu
    ```

=== "GPU with memory limits"
    ```bash
    # Process smaller batches if GPU memory is limited
    gfpgan-infer --input photos/ --device cuda --bg_tile 200
    ```

## Quality tracking and metrics

### Enable quality metrics

=== "Fast metrics (LPIPS only)"
    ```bash
    gfpgan-infer --input photos/ --metrics fast --report-path quick_report.json
    ```

=== "Detailed metrics (LPIPS, DISTS, ArcFace)"
    ```bash
    gfpgan-infer --input photos/ --metrics detailed --report-path full_report.json
    ```

=== "No metrics (fastest)"
    ```bash
    gfpgan-infer --input photos/ --metrics none
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
gfpgan-infer --input photos/ --deterministic --random-seed 42
```

### Preserve metadata

Keep original EXIF data:

```bash
gfpgan-infer --input photos/ --preserve-metadata
```

### Provenance tracking

GFPGAN automatically saves processing information:

```
results/
├── restored_photos/         # Processed images
├── batch_report.json       # Quality metrics
├── processing_log.txt      # Detailed log
└── provenance.json         # Settings and environment info
```

## Monitoring progress

### Progress indicators

=== "Simple progress bar"
    ```bash
    gfpgan-infer --input photos/ --progress
    ```

=== "Detailed logging"
    ```bash
    gfpgan-infer --input photos/ --verbose --log-file batch.log
    ```

=== "Silent mode"
    ```bash
    gfpgan-infer --input photos/ --quiet
    ```

### Handling interruptions

If processing is interrupted:

```bash
# Resume from where it left off
gfpgan-infer --input photos/ --resume --checkpoint-dir .checkpoints/
```

## Presets and automation

### Create processing presets

Save commonly used settings:

=== "High quality preset"
    ```bash
    # Create alias or script
    alias gfpgan-hq='gfpgan-infer --version 1.4 --bg_upsampler realesrgan --metrics detailed'

    # Use the preset
    gfpgan-hq --input photos/ --output results/
    ```

=== "Fast processing preset"
    ```bash
    alias gfpgan-fast='gfpgan-infer --backend codeformer --bg_upsampler none --metrics fast'

    gfpgan-fast --input photos/ --output results/
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

    gfpgan-infer \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR" \
        --version 1.4 \
        --metrics detailed \
        --progress \
        --report-path "$OUTPUT_DIR/quality_report.json"
    ```

=== "Python script example"
    ```python
    #!/usr/bin/env python3
    import subprocess
    import sys
    from pathlib import Path

    def batch_restore(input_dir, output_dir, backend="gfpgan", version="1.4"):
        cmd = [
            "gfpgan-infer",
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--backend", backend,
            "--version", version,
            "--metrics", "detailed",
            "--progress"
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
    gfpgan-infer --input photos/ --device cpu

    # Reduce background tile size
    gfpgan-infer --input photos/ --bg_tile 200

    # Disable background enhancement
    gfpgan-infer --input photos/ --bg_upsampler none
    ```

!!! warning "Some images failed to process"
    Check the processing log:
    ```bash
    gfpgan-infer --input photos/ --log-file processing.log --verbose
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
