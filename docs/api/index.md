# API Documentation

GFPGAN provides both REST API and Python API for integrating face restoration into your applications.

## REST API

### Auto-generated Documentation

GFPGAN includes a FastAPI-based REST API with automatic documentation:

- **Interactive docs**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)
- **Alternative docs**: [http://localhost:8000/redoc](http:For implementation examples and advanced usage patterns, see our [user guides](../guides/face-enhancement.md)./localhost:8000/redoc) (ReDoc)
- **OpenAPI spec**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### Starting the API Server

```bash
# Production server
uvicorn services.api.main:app --host 0.0.0.0 --port 8000

# Development server with auto-reload
uvicorn services.api.main:app --reload --port 8000
```

### Authentication

Currently, the API runs without authentication for simplicity. For production deployments, implement authentication:

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

## Python API

### Core Classes

#### GFPGANer

Main restoration class for processing images:

```python
from gfpgan import GFPGANer
import cv2

# Initialize restorer
restorer = GFPGANer(
    model_path='experiments/pretrained_models/GFPGANv1.4.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None  # or 'realesrgan'
)

# Restore image
input_img = cv2.imread('input.jpg')
cropped_faces, restored_imgs, restored_faces = restorer.enhance(
    input_img,
    has_aligned=False,
    only_center_face=False,
    paste_back=True
)

# Save result
cv2.imwrite('output.jpg', restored_imgs[0])
```

#### Key Parameters

- **model_path**: Path to GFPGAN model file
- **upscale**: Upscaling factor (1, 2, or 4)
- **arch**: Model architecture ('clean', 'original')
- **channel_multiplier**: Channel multiplier for StyleGAN decoder
- **bg_upsampler**: Background upsampler ('realesrgan', 'esrgan', None)

### Utility Functions

#### Image Processing

```python
from gfpgan.utils import restore_image

# Simple restoration function
restored_img = restore_image(
    image_path='input.jpg',
    output_path='output.jpg',
    version='1.4',
    upscale=2
)
```

#### Model Management

```python
from gfpgan.utils import download_model, list_models

# Download model if not present
model_path = download_model('GFPGANv1.4')

# List available models
models = list_models()
print(models)  # ['GFPGANv1.3', 'GFPGANv1.4', 'RestoreFormer++']
```

#### Quality Metrics

```python
from gfpgan.metrics import calculate_metrics

# Calculate quality metrics
metrics = calculate_metrics(
    original_img='original.jpg',
    restored_img='restored.jpg',
    metrics=['lpips', 'dists', 'arcface']
)

print(f"LPIPS: {metrics['lpips']:.3f}")
print(f"DISTS: {metrics['dists']:.3f}")
print(f"ArcFace Similarity: {metrics['arcface']:.3f}")
```

### Advanced Usage

#### Custom Model Configuration

```python
from gfpgan import GFPGANer

# Custom model configuration
restorer = GFPGANer(
    model_path='path/to/custom_model.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler='realesrgan',
    device='cuda',  # or 'cpu', 'mps'
    model_root_path='experiments/pretrained_models'
)

# Process with custom settings
output = restorer.enhance(
    img=input_image,
    has_aligned=False,
    only_center_face=False,
    paste_back=True,
    weight=0.5  # Blend weight with original
)
```

#### Batch Processing

```python
import os
from pathlib import Path
from gfpgan import GFPGANer

restorer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=2)

# Process directory
input_dir = Path('input_images')
output_dir = Path('restored_images')
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob('*.jpg'):
    img = cv2.imread(str(img_path))
    _, restored_imgs, _ = restorer.enhance(img)

    output_path = output_dir / f"restored_{img_path.name}"
    cv2.imwrite(str(output_path), restored_imgs[0])
    print(f"Processed: {img_path.name}")
```

#### Error Handling

```python
from gfpgan import GFPGANer
from gfpgan.exceptions import GFPGANError, ModelNotFoundError

try:
    restorer = GFPGANer(model_path='invalid_model.pth')
except ModelNotFoundError:
    print("Model not found, downloading...")
    # Handle model download

try:
    result = restorer.enhance(damaged_image)
except GFPGANError as e:
    print(f"Restoration failed: {e}")
    # Handle restoration failure
```

## Integration Examples

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

**Need help?** Check our [guides](../guides/) or [create an issue](https://github.com/IAmJonoBo/GFPGAN/issues).
