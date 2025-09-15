# Restore a photo

Learn how to restore a single image using GFPGAN's CLI and web interface.

## Quick start

For a damaged or blurry photo, restoration takes just one command:

```bash
gfpgan-infer --input damaged_photo.jpg --version 1.4
```

Results are saved to `results/` with before/after comparison images.

## CLI workflow

### Basic restoration

=== "Single image"
    ```bash
    gfpgan-infer --input photo.jpg --version 1.4
    ```

=== "Custom output location"
    ```bash
    gfpgan-infer --input photo.jpg --output restored/ --version 1.4
    ```

=== "Upscale while restoring"
    ```bash
    gfpgan-infer --input photo.jpg --version 1.4 --upscale 2
    ```

### Choose your backend

Different backends offer different trade-offs:

=== "GFPGAN v1.4 (recommended)"
    ```bash
    gfpgan-infer --input photo.jpg --version 1.4
    ```
    - **Best for**: General photos and portraits
    - **Quality**: High detail preservation
    - **Speed**: Medium

=== "GFPGAN v1.3 (natural)"
    ```bash
    gfpgan-infer --input photo.jpg --version 1.3
    ```
    - **Best for**: Natural-looking results
    - **Quality**: Good, less sharp than v1.4
    - **Speed**: Medium

=== "CodeFormer (fast)"
    ```bash
    gfpgan-infer --input photo.jpg --backend codeformer
    ```
    - **Best for**: Batch processing
    - **Quality**: Good
    - **Speed**: Fast

### Advanced options

=== "Background enhancement"
    ```bash
    # Enhance background with Real-ESRGAN
    gfpgan-infer --input photo.jpg --bg_upsampler realesrgan

    # Disable background enhancement for speed
    gfpgan-infer --input photo.jpg --bg_upsampler none
    ```

=== "Face detection options"
    ```bash
    # Only restore the center face
    gfpgan-infer --input photo.jpg --only_center_face

    # Input is already an aligned face crop
    gfpgan-infer --input face_crop.jpg --aligned
    ```

=== "Device selection"
    ```bash
    # Force CPU (if GPU has issues)
    gfpgan-infer --input photo.jpg --device cpu

    # Auto-detect best device
    gfpgan-infer --input photo.jpg --device auto
    ```

## Web interface

Launch the interactive web UI for drag-and-drop restoration:

```bash
python -m gfpgan.gradio_app
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Web interface features

- **Drag and drop**: Upload images directly
- **Real-time preview**: See results before saving
- **Parameter adjustment**: Change models and settings interactively
- **Download options**: Get individual results or batch ZIP

### Custom web server

For more control, use the FastAPI server:

```bash
uvicorn services.api.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation available at:
- Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc format: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Understanding results

### Output structure

After running restoration, you'll find:

```
results/
├── restored_faces/           # Individual face crops (restored)
├── cropped_faces/           # Original face crops (input)
├── cmp_photo.jpg           # Before/after comparison
└── photo.jpg               # Full restored image
```

### Quality assessment

Visual indicators to check:
- **Sharpness**: Are facial features well-defined?
- **Identity preservation**: Does the person still look like themselves?
- **Artifacts**: Any unnatural textures or distortions?
- **Background**: Is the non-face area properly enhanced?

## Troubleshooting

### Common issues

!!! warning "No faces detected"
    ```
    Warning: No faces detected in the input image.
    ```
    **Solutions:**
    - Ensure the image contains visible faces
    - Try a different face detection threshold
    - Check if the image is too small or low quality

!!! error "CUDA out of memory"
    ```
    RuntimeError: CUDA out of memory
    ```
    **Solutions:**
    ```bash
    # Use CPU instead
    gfpgan-infer --input photo.jpg --device cpu

    # Reduce image size first
    gfpgan-infer --input photo.jpg --upscale 1
    ```

!!! warning "Poor restoration quality"
    **Try different backends:**
    ```bash
    # More natural results
    gfpgan-infer --input photo.jpg --version 1.3

    # Better identity preservation
    gfpgan-infer --input photo.jpg --backend restoreformer
    ```

### Getting help

If restoration quality isn't satisfactory:

1. Check our [backend comparison guide](choose-backend.md)
2. Review [quality metrics](metrics.md) for objective evaluation
3. See [troubleshooting](../troubleshooting.md) for technical issues

---

**Next steps:**
- [Process multiple photos →](batch-processing.md)
- [Measure restoration quality →](metrics.md)
- [Choose the right backend →](choose-backend.md)
