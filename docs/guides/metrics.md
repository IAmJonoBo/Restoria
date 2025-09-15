# Quality metrics

Measure restoration quality objectively with built-in metrics and learn how to interpret results.

## Quick start

Generate a quality report for your restored images:

```bash
gfpgan-infer --input photos/ --metrics detailed --report-path quality_report.json
```

This creates a comprehensive JSON report with quality scores for each image.

## Available metrics

### LPIPS (Learned Perceptual Image Patch Similarity)

**What it measures:** Perceptual similarity between original and restored images

- **Range:** 0.0 to 1.0
- **Lower is better** (0.0 = identical, 1.0 = very different)
- **Typical good range:** 0.1 to 0.3

```bash
gfpgan-infer --input photo.jpg --metrics lpips
```

**Interpretation:**
- `< 0.2`: Excellent perceptual quality
- `0.2 - 0.3`: Good quality
- `0.3 - 0.4`: Acceptable quality
- `> 0.4`: Poor quality

### DISTS (Deep Image Structure and Texture Similarity)

**What it measures:** Structural and texture similarity

- **Range:** 0.0 to 1.0
- **Lower is better**
- **Typical good range:** 0.1 to 0.25

```bash
gfpgan-infer --input photo.jpg --metrics dists
```

**Interpretation:**
- `< 0.15`: Excellent structural preservation
- `0.15 - 0.25`: Good structural quality
- `0.25 - 0.35`: Acceptable quality
- `> 0.35`: Poor structural preservation

### ArcFace Identity Similarity

**What it measures:** How well facial identity is preserved

- **Range:** 0.0 to 1.0
- **Higher is better** (1.0 = perfect identity match)
- **Typical good range:** 0.7 to 0.95

```bash
gfpgan-infer --input photo.jpg --metrics arcface
```

**Interpretation:**
- `> 0.9`: Excellent identity preservation
- `0.8 - 0.9`: Good identity preservation
- `0.7 - 0.8`: Acceptable identity preservation
- `< 0.7`: Poor identity preservation

## Metric presets

### Fast metrics (recommended for batch)

```bash
gfpgan-infer --input photos/ --metrics fast
```

**Includes:** LPIPS only
**Processing time:** ~10% overhead
**Best for:** Large batches, quick quality checks

### Detailed metrics (comprehensive)

```bash
gfpgan-infer --input photos/ --metrics detailed
```

**Includes:** LPIPS, DISTS, ArcFace, processing time
**Processing time:** ~30% overhead
**Best for:** Quality analysis, backend comparison, research

### No metrics (fastest)

```bash
gfpgan-infer --input photos/ --metrics none
```

**Includes:** Processing time only
**Best for:** Production workflows where speed matters most

## Understanding reports

### JSON report structure

```json
{
    "summary": {
        "total_images": 50,
        "successful": 48,
        "failed": 2,
        "avg_lpips": 0.187,
        "avg_dists": 0.142,
        "avg_arcface": 0.863,
        "avg_processing_time": 2.34,
        "backend": "gfpgan_v1.4",
        "timestamp": "2024-01-15T10:30:00Z"
    },
    "per_image": {
        "family_photo_001.jpg": {
            "lpips": 0.156,
            "dists": 0.128,
            "arcface_similarity": 0.891,
            "processing_time": 2.1,
            "faces_detected": 2,
            "status": "success"
        },
        "portrait_002.jpg": {
            "lpips": 0.203,
            "dists": 0.167,
            "arcface_similarity": 0.834,
            "processing_time": 2.6,
            "faces_detected": 1,
            "status": "success"
        }
    },
    "failed_images": {
        "corrupted_image.jpg": {
            "error": "No faces detected",
            "status": "failed"
        }
    }
}
```

### Key report sections

=== "Summary statistics"
    - **Total/successful/failed counts**
    - **Average metric scores**
    - **Overall processing performance**
    - **Backend and settings used**

=== "Per-image details"
    - **Individual quality scores**
    - **Number of faces detected**
    - **Processing time per image**
    - **Success/failure status**

=== "Failure analysis"
    - **Failed image list**
    - **Error reasons**
    - **Troubleshooting hints**

## Comparing backends

### Benchmark multiple backends

```bash
# Test GFPGAN v1.4
gfpgan-infer --input test_photos/ --version 1.4 --metrics detailed --output results_v14/ --report-path v14_report.json

# Test CodeFormer
gfpgan-infer --input test_photos/ --backend codeformer --metrics detailed --output results_cf/ --report-path cf_report.json

# Test RestoreFormer++
gfpgan-infer --input test_photos/ --backend restoreformer --metrics detailed --output results_rf/ --report-path rf_report.json
```

### Analyze backend performance

Create a comparison script:

```python
import json
import pandas as pd

def compare_backends(report_files, backend_names):
    results = []

    for report_file, backend in zip(report_files, backend_names):
        with open(report_file) as f:
            data = json.load(f)

        results.append({
            'backend': backend,
            'avg_lpips': data['summary']['avg_lpips'],
            'avg_dists': data['summary']['avg_dists'],
            'avg_arcface': data['summary']['avg_arcface'],
            'avg_time': data['summary']['avg_processing_time'],
            'success_rate': data['summary']['successful'] / data['summary']['total_images']
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

# Usage
compare_backends(
    ['v14_report.json', 'cf_report.json', 'rf_report.json'],
    ['GFPGAN v1.4', 'CodeFormer', 'RestoreFormer++']
)
```

## Quality thresholds and recommendations

### Production quality gates

Set quality thresholds for automated workflows:

```python
def quality_check(report_path, min_lpips=0.3, min_arcface=0.7):
    with open(report_path) as f:
        data = json.load(f)

    passed = []
    failed = []

    for filename, metrics in data['per_image'].items():
        if (metrics['lpips'] <= min_lpips and
            metrics['arcface_similarity'] >= min_arcface):
            passed.append(filename)
        else:
            failed.append(filename)

    return passed, failed
```

### Recommended thresholds by use case

=== "Professional/Commercial"
    - **LPIPS**: < 0.2
    - **DISTS**: < 0.15
    - **ArcFace**: > 0.85

    ```bash
    gfpgan-infer --input photos/ --backend restoreformer --metrics detailed
    ```

=== "Social Media/Personal"
    - **LPIPS**: < 0.3
    - **DISTS**: < 0.25
    - **ArcFace**: > 0.75

    ```bash
    gfpgan-infer --input photos/ --version 1.4 --metrics fast
    ```

=== "Archive/Historical"
    - **LPIPS**: < 0.4
    - **DISTS**: < 0.35
    - **ArcFace**: > 0.65

    ```bash
    gfpgan-infer --input photos/ --version 1.3 --metrics detailed
    ```

## Advanced metrics analysis

### Statistical analysis

```python
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_metrics(report_path):
    with open(report_path) as f:
        data = json.load(f)

    lpips_scores = [img['lpips'] for img in data['per_image'].values()]
    arcface_scores = [img['arcface_similarity'] for img in data['per_image'].values()]

    print(f"LPIPS - Mean: {np.mean(lpips_scores):.3f}, Std: {np.std(lpips_scores):.3f}")
    print(f"ArcFace - Mean: {np.mean(arcface_scores):.3f}, Std: {np.std(arcface_scores):.3f}")

    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(lpips_scores, bins=20, alpha=0.7)
    ax1.set_xlabel('LPIPS Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('LPIPS Distribution')

    ax2.hist(arcface_scores, bins=20, alpha=0.7)
    ax2.set_xlabel('ArcFace Similarity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Identity Preservation Distribution')

    plt.tight_layout()
    plt.savefig('metrics_analysis.png')
```

### Correlation analysis

```python
def correlation_analysis(report_path):
    with open(report_path) as f:
        data = json.load(f)

    metrics_data = []
    for img_data in data['per_image'].values():
        metrics_data.append([
            img_data['lpips'],
            img_data['dists'],
            img_data['arcface_similarity'],
            img_data['processing_time'],
            img_data['faces_detected']
        ])

    df = pd.DataFrame(metrics_data,
                     columns=['LPIPS', 'DISTS', 'ArcFace', 'Time', 'Faces'])

    correlation_matrix = df.corr()
    print(correlation_matrix)
```

## Troubleshooting metrics

### Common issues

!!! warning "Metrics computation failed"
    ```
    Error: Failed to compute ArcFace similarity
    ```
    **Solutions:**
    - Install required dependencies: `pip install -e ".[metrics]"`
    - Check if faces were detected in the image
    - Try with `--metrics fast` (LPIPS only)

!!! info "Unexpected metric values"
    **Very high LPIPS (> 0.5):**
    - Check if input/output images are properly aligned
    - Verify the restoration actually improved the image
    - Consider trying a different backend

    **Very low ArcFace (< 0.5):**
    - Face detection might have failed
    - Original image quality might be too poor
    - Identity may have been significantly altered

!!! tip "Performance considerations"
    **Metrics slow down processing:**
    - Use `--metrics fast` for large batches
    - Consider `--metrics none` for production workflows
    - Process metrics separately on a subset for quality assessment

---

**Next steps:**
- [Choose the right backend →](choose-backend.md)
- [Optimize hardware performance →](../HARDWARE_GUIDE.md)
- [Set up batch processing →](batch-processing.md)
