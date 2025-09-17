# Restore a photo

Learn how to restore a single image using GFPGAN's CLI and web interface.

## Quick start

For a damaged or blurry photo, restoration takes just one command:

```bash
restoria run --input damaged_photo.jpg --backend gfpgan
```

Results are saved to `results/` with before/after comparison images.

## CLI workflow

### Basic restoration

=== "Single image"
    ```bash
    restoria run --input photo.jpg --backend gfpgan
    ```

=== "Custom output location"
    ```bash
    restoria run --input photo.jpg --output restored/ --backend gfpgan
    ```

=== "Upscale while restoring"
    ```bash
    restoria run --input photo.jpg --backend gfpgan --output out/
    ```

### Choose your backend

Different backends offer different trade-offs:

=== "GFPGAN v1.4 (recommended)"
    ```bash
    restoria run --input photo.jpg --backend gfpgan
    ```
    - **Best for**: General photos and portraits
    - **Quality**: High detail preservation
    - **Speed**: Medium

=== "GFPGAN v1.3 (natural)"
    ```bash
    restoria run --input photo.jpg --backend gfpgan
    ```
    - **Best for**: Natural-looking results
    - **Quality**: Good, less sharp than v1.4
    - **Speed**: Medium

=== "CodeFormer (fast)"
    ```bash
    restoria run --input photo.jpg --backend codeformer
    ```
    - **Best for**: Batch processing
    - **Quality**: Good
    - **Speed**: Fast

### Advanced options

=== "Background enhancement"
    ```bash
    # Restore with GFPGAN backend
    restoria run --input photo.jpg --backend gfpgan
    ```

=== "Face detection options"
    ```bash
    # Single-image restoration
    restoria run --input photo.jpg --backend gfpgan
    ```

### Device selection

```bash
# Force CPU (if GPU has issues)
restoria run --input photo.jpg --device cpu --backend codeformer

# Auto-detect best device
restoria run --input photo.jpg --device auto --backend gfpgan
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

```text
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
    A common message during face detection.

```text
Warning: No faces detected in the input image.
```

**Solutions:**

- Ensure the image contains visible faces
- Try a different face detection threshold
- Check if the image is too small or low quality

!!! error "CUDA out of memory"
    Insufficient GPU memory for the selected settings.

```text
RuntimeError: CUDA out of memory
```

**Solutions:**

```bash
# Use CPU instead
restoria run --input photo.jpg --device cpu --backend codeformer

# Reduce processing load
restoria run --input photo.jpg --backend codeformer --metrics off
```

!!! warning "Poor restoration quality"
    If results look off, try an alternative backend.

**Try different backends:**

```bash
# Natural results
restoria run --input photo.jpg --backend gfpgan

# Better identity preservation
restoria run --input photo.jpg --backend restoreformerpp
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
