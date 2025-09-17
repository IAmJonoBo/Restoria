# Choose backend

Compare restoration backends and select the best one for your needs.

## Backend comparison matrix

| Backend | Speed | Quality | Identity | Memory | Best for |
|---------|-------|---------|----------|--------|----------|
| **GFPGAN v1.4** | Medium | High | Excellent | Medium | General photos |
| **GFPGAN v1.3** | Medium | High | Good | Medium | Natural results |
| **GFPGAN v1.2** | Medium | Good | Good | Medium | Sharp details |
| **CodeFormer** | Fast | Medium | Good | Low | Batch processing |
| **RestoreFormer++** | Slow | Highest | Excellent | High | Professional work |

## Detailed comparisons

### GFPGAN models

=== "GFPGAN v1.4 (Recommended)"
    ```bash
    # New CLI
    gfpup run --input photo.jpg --backend gfpgan --output out/

    # Legacy shim
    gfpgan-infer --input photo.jpg --version 1.4
    ```

    **Strengths:**
    - Excellent detail preservation
    - Good identity preservation
    - Works well on various photo types
    - Balanced speed/quality trade-off

    **Best for:**
    - General portrait restoration
    - Mixed photo collections
    - Production workflows

    **Example use case:** Family photo restoration, professional headshots

=== "GFPGAN v1.3 (Natural)"
    ```bash
    # New CLI (select model via param if supported)
    gfpup run --input photo.jpg --backend gfpgan --output out/

    # Legacy shim
    gfpgan-infer --input photo.jpg --version 1.3
    ```

    **Strengths:**
    - More natural-looking results
    - Less artificial sharpening
    - Good for low-quality inputs
    - Handles repeated restoration well

    **Weaknesses:**
    - Slightly less sharp than v1.4
    - Some identity changes possible

    **Best for:**
    - Natural photo restoration
    - Social media photos
    - When subtlety is preferred

    **Example use case:** Old family photos, vintage portraits

=== "GFPGAN v1.2 (Sharp)"
    ```bash
    # Legacy shim example only
    gfpgan-infer --input photo.jpg --version 1.2
    ```

    **Strengths:**
    - Very sharp output
    - Good detail enhancement
    - Suitable for beauty/makeup photos

    **Weaknesses:**
    - Can look unnatural
    - May over-enhance features

    **Best for:**
    - Beauty photography
    - When maximum sharpness is needed
    - Fashion/glamour photos

### Alternative backends

=== "CodeFormer (Fast)"
    ```bash
    # New CLI
    gfpup run --input photo.jpg --backend codeformer --output out/

    # Legacy shim
    gfpgan-infer --input photo.jpg --backend codeformer
    ```

    **Strengths:**
    - Fastest processing
    - Lower memory usage
    - Good identity preservation
    - Controllable restoration strength

    **Weaknesses:**
    - Lower detail quality than GFPGAN
    - Less sophisticated texture handling

    **Best for:**
    - Large batch processing
    - Resource-constrained environments
    - Quick previews

    **Example use case:** Processing thousands of photos, real-time applications

=== "RestoreFormer++ (Premium)"
    ```bash
    # New CLI uses canonical name restoreformerpp
    gfpup run --input photo.jpg --backend restoreformerpp --output out/

    # Legacy shim alias
    gfpgan-infer --input photo.jpg --backend restoreformer
    ```

    **Strengths:**
    - Highest quality results
    - Excellent identity preservation
    - Superior texture reconstruction
    - Advanced face parsing

    **Weaknesses:**
    - Slowest processing
    - Highest memory usage
    - May require more powerful hardware

    **Best for:**
    - Professional photo restoration
    - High-value images
    - When quality is paramount

    **Example use case:** Archive restoration, professional photography, art restoration

## Selection guidelines

### By use case

=== "Personal photos"
    **Recommended:** GFPGAN v1.4
    ```bash
    gfpgan-infer --input family_photo.jpg --version 1.4
    ```
    - Good balance of quality and speed
    - Preserves natural appearance
    - Handles various lighting conditions

=== "Professional work"
    **Recommended:** RestoreFormer++
    ```bash
        gfpup run \
            --input portrait.jpg \
            --backend restoreformerpp \
            --metrics full \
            --output out/
    ```
    - Highest quality output
    - Excellent for client work
    - Detailed quality metrics

=== "Batch processing"
    **Recommended:** CodeFormer
    ```bash
    gfpup run --input photos/ --backend codeformer --output out/
    ```
    - Fastest processing
    - Lower resource usage
    - Still produces good results

=== "Archive restoration"
    **Recommended:** GFPGAN v1.3 or RestoreFormer++
    ```bash
    # For natural results
    gfpgan-infer --input old_photo.jpg --version 1.3

    # For maximum quality
    gfpup run --input old_photo.jpg --backend restoreformerpp --output out/
    ```

### By hardware

=== "High-end GPU (8GB+ VRAM)"
    - ✅ All backends supported
    - **Recommended:** RestoreFormer++ for quality
    - **Alternative:** GFPGAN v1.4 for speed

    ```bash
    gfpup run --input photo.jpg --backend restoreformerpp --output out/ --compile
    ```

=== "Mid-range GPU (4-8GB VRAM)"
    - ✅ GFPGAN (all versions)
    - ✅ CodeFormer
    - ⚠️ RestoreFormer++ (may need CPU fallback)

    ```bash
    gfpgan-infer --input photo.jpg --version 1.4 --bg_tile 400
    ```

=== "Low-end GPU (<4GB VRAM)"
    - ✅ CodeFormer
    - ✅ GFPGAN with reduced settings
    - ❌ RestoreFormer++ (use CPU)

    ```bash
    gfpup run --input photo.jpg --backend codeformer --output out/
    ```

=== "CPU only"
    - ✅ All backends (slower)
    - **Recommended:** CodeFormer for speed
    - Disable background enhancement

    ```bash
    gfpup run --input photo.jpg --device cpu --backend codeformer --output out/
    ```

## Performance benchmarks

### Processing speed (avg. per image)

| Backend | GPU (RTX 3080) | CPU (Intel i7) |
|---------|----------------|----------------|
| CodeFormer | 0.8s | 12s |
| GFPGAN v1.4 | 1.2s | 18s |
| GFPGAN v1.3 | 1.1s | 17s |
| RestoreFormer++ | 2.4s | 45s |

### Memory usage

| Backend | GPU Memory | System RAM |
|---------|------------|------------|
| CodeFormer | 2GB | 4GB |
| GFPGAN v1.4 | 3GB | 6GB |
| GFPGAN v1.3 | 3GB | 6GB |
| RestoreFormer++ | 5GB | 8GB |

Note: benchmarks based on 512x512 input images with background enhancement.

## Quality evaluation

### Objective metrics

Use built-in metrics to compare backends:

        # Test multiple backends on the same image
        gfpup run --input test_photo.jpg \
            --backend gfpgan --metrics full --output v14/
        gfpup run --input test_photo.jpg \
            --backend codeformer --metrics full --output cf/
        gfpup run \
            --input test_photo.jpg \
            --backend restoreformerpp \
            --metrics full \
            --output rf/

**Typical metric ranges:**

- **LPIPS**: 0.1-0.4 (lower is better)
- **DISTS**: 0.1-0.3 (lower is better)
- **ArcFace**: 0.7-0.95 (higher is better)

### Visual quality checklist

When comparing results, look for:

- **Sharpness**: Are facial features well-defined?
- **Naturalness**: Does the person look realistic?
- **Identity**: Is the person still recognizable?
- **Artifacts**: Any unnatural textures or distortions?
- **Consistency**: Similar quality across multiple faces?

## Advanced configuration

### Optional ensemble (experimental)

You can blend outputs from multiple backends by selecting the ensemble backend.
This is off by default and requires no extra installs for a simple blend.

Example:

        gfpup run --input photo.jpg \
            --backend ensemble \
            --ensemble-backends gfpgan,codeformer \
            --ensemble-weights 0.5,0.5 \
            --output out/

Missing backends are skipped gracefully and the run proceeds.

### Backend-specific parameters

=== "GFPGAN tuning"
    ```bash
    # Adjust upsampling
    gfpgan-infer --input photo.jpg --version 1.4 --upscale 1  # No upsampling
    gfpgan-infer --input photo.jpg --version 1.4 --upscale 4  # 4x upsampling

    # Background tile size (affects memory)
    gfpgan-infer --input photo.jpg --version 1.4 --bg_tile 200  # Smaller tiles
    ```

=== "CodeFormer tuning"
    ```bash
    # Restoration strength (if supported)
    gfpgan-infer --input photo.jpg --backend codeformer --fidelity 0.8

    # Face detection threshold
    gfpgan-infer --input photo.jpg --backend codeformer --detection_threshold 0.5
    ```

=== "RestoreFormer++ tuning"
    ```bash
    # High quality mode
    gfpgan-infer --input photo.jpg --backend restoreformer --high_quality

    # Memory optimization
    gfpgan-infer --input photo.jpg --backend restoreformer --memory_efficient
    ```

## Decision flowchart

        Start
            ↓
        Quality most important? → Yes → Use RestoreFormer++
            ↓ No
        Speed most important? → Yes → Use CodeFormer
            ↓ No
        Natural results preferred? → Yes → Use GFPGAN v1.3
            ↓ No
        General use → Use GFPGAN v1.4

---

**Next steps:**

- [Measure quality with metrics →](metrics.md)
- [Process multiple photos →](batch-processing.md)
- [Optimize hardware performance →](../HARDWARE_GUIDE.md)
