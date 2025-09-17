# API Documentation

This project provides two integration surfaces:

- REST API (FastAPI) for HTTP-based usage
- Python API via the new modular layer under `src/gfpp/` (preferred), with
    a legacy GFPGAN API kept for backward compatibility

## REST API

### Auto-generated Documentation

GFPGAN includes a FastAPI-based REST API with automatic documentation:

- **Interactive docs**:
    [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)
- **Alternative docs**:
    [http://localhost:8000/redoc](http://localhost:8000/redoc) (ReDoc)
- **OpenAPI spec**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### Starting the API Server

```bash
# Production server
uvicorn services.api.main:app --host 0.0.0.0 --port 8000

# Development server with auto-reload
uvicorn services.api.main:app --reload --port 8000
```

### Authentication

Currently, the API runs without authentication for simplicity.
For production deployments, implement authentication:

```python
# Example: API key authentication
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

def verify_api_key(token: str = Depends(security)):
    if token.credentials != "your-api-key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return token
```

### Example API Usage

#### Single Image Restoration

**Request:**

```bash
curl -X POST "http://localhost:8000/restore" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@damaged_photo.jpg" \
  -F "version=1.4" \
  -F "upscale=2"
```

**Response:**

```json
{
  "status": "success",
  "restoration_id": "12345-abcde",
  "original_size": [800, 600],
  "restored_size": [1600, 1200],
  "processing_time": 2.3,
  "download_url": "/download/12345-abcde"
}
```

#### Batch Processing

**Request:**

```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": [
      "https://example.com/photo1.jpg",
      "https://example.com/photo2.jpg"
    ],
    "settings": {
      "version": "1.4",
      "upscale": 2,
      "background_enhance": true
    }
  }'
```

**Response:**

```json
{
  "status": "accepted",
  "batch_id": "batch-67890",
  "estimated_completion": "2024-01-15T10:30:00Z",
  "status_url": "/batch/batch-67890/status"
}
```

#### Quality Metrics

**Request:**

```bash
curl -X POST "http://localhost:8000/metrics" \
  -H "Content-Type: multipart/form-data" \
  -F "original=@original.jpg" \
  -F "restored=@restored.jpg" \
  -F "metrics[]=lpips" \
  -F "metrics[]=dists"
```

**Response:**

```json
{
  "metrics": {
    "lpips": 0.234,
    "dists": 0.156,
    "arcface_similarity": 0.892
  },
  "analysis": {
    "quality_score": 8.5,
    "identity_preservation": "excellent",
    "texture_realism": "high"
  }
}
```

## Python API (gfpp)

Prefer the modular `gfpp` API for new integrations. It exposes:

- Orchestrator: produce a deterministic Plan for an image and options
- Registry: list/resolve backends without importing heavy deps
- IO: run manifest helpers and centralized weight resolution
- Restorers: lightweight adapters that implement a shared protocol

### Orchestrator and Plan

```python
from gfpp.core.orchestrator import plan

pl = plan(
    "samples/portrait.jpg",
    {"backend": "gfpgan", "weight": 0.6, "experimental": False},
)
print(pl.backend, pl.params, pl.reason, pl.confidence)
print(pl.quality)  # may include niqe/brisque (best-effort)
```

Plan fields (dataclass):

- backend: str
- params: dict[str, any]
- postproc: dict[str, any]
- reason: str
- confidence: float (0..1)
- quality: dict[str, float | None]
- faces: dict[str, any]
- detail: dict[str, any] (routing explanation and inputs)

### Registry

```python
from gfpp.core.registry import list_backends, get

avail = list_backends()           # {'gfpgan': True, 'codeformer': False, ...}
RestorerClass = get('gfpgan')     # returns a class, import deferred until here
```

### Restorer Protocol

```python
from gfpp.restorers.base import Restorer, RestoreResult

def run_restoration(img):
    RestorerClass = get('gfpgan')
    restorer: Restorer = RestorerClass()
    restorer.prepare({"device": "auto"})  # lazy import heavy deps
    result: RestoreResult = restorer.restore(img, {"weight": 0.6})
    return result.restored_image, result.metrics
```

RestoreResult fields:

- input_path: str | None
- restored_path: str | None
- restored_image: any | None (e.g., numpy.ndarray)
- cropped_faces: list[str]
- restored_faces: list[str]
- metrics: dict[str, any]

### IO helpers

```python
from gfpp.io.manifest import RunManifest, write_manifest
from gfpp.io.weights import ensure_weight

man = RunManifest(args={"backend": "gfpgan", "metrics": "fast"}, device="cpu")
ensure_weight("GFPGANv1.4")  # delegates to centralized resolver
write_manifest("out/manifest.json", man)
```

## Integration Examples (legacy GFPGAN API)

### Flask Application

```python
from flask import Flask, request, send_file
from gfpgan import GFPGANer
import cv2
import tempfile

app = Flask(__name__)
restorer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=2)

@app.route('/restore', methods=['POST'])
def restore_image():
    file = request.files['image']

    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_input:
        file.save(tmp_input.name)
        img = cv2.imread(tmp_input.name)

        # Restore image
        _, restored_imgs, _ = restorer.enhance(img)

        # Save result
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_output:
            cv2.imwrite(tmp_output.name, restored_imgs[0])
            return send_file(tmp_output.name, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
```

### Streamlit Application

```python
import streamlit as st
from gfpgan import GFPGANer
import cv2
import numpy as np

@st.cache_resource
def load_restorer():
    return GFPGANer(model_path='GFPGANv1.4.pth', upscale=2)

st.title("GFPGAN Face Restoration")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    with col2:
        st.header("Restored")
        restorer = load_restorer()
        _, restored_imgs, _ = restorer.enhance(image)
        st.image(cv2.cvtColor(restored_imgs[0], cv2.COLOR_BGR2RGB))
```

## Performance Optimization

### GPU Acceleration

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Use specific device
restorer = GFPGANer(
    model_path='GFPGANv1.4.pth',
    device='cuda:0'  # or 'cpu', 'mps'
)
```

### Memory Management

```python
# For processing large images or batches
restorer = GFPGANer(
    model_path='GFPGANv1.4.pth',
    upscale=1,  # Reduce upscaling to save memory
    bg_upsampler=None  # Disable background upsampling
)

# Clear GPU cache after processing
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| 400 | Invalid input format | Check image format and size |
| 404 | Model not found | Verify model path or download |
| 413 | Image too large | Reduce image size or increase limits |
| 500 | Processing failed | Check GPU memory and input validity |
| 503 | Service overloaded | Implement rate limiting or retry |

---

**Need help?** Check our [guides](../guides/face-enhancement.md) or [create an issue](https://github.com/IAmJonoBo/Restoria/issues).
