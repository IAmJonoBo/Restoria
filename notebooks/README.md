# Notebooks

<!-- markdownlint-disable MD013 -->

This directory contains Jupyter notebooks for interactive restoration with Restoria (GFPGAN-compatible).

## Available Notebooks

### Restoria_Colab.ipynb

Interactive Google Colab notebook for easy experimentation with Restoria:

- **Purpose**: Quick start guide and interactive demos
- **Platform**: Google Colab (can also run locally)
- **Features**:
  - Step-by-step restoration workflow
  - Before/after comparisons
  - Multiple model comparisons
  - Quality metrics evaluation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IAmJonoBo/Restoria/blob/main/notebooks/Restoria_Colab.ipynb)

Additional Restoria notebooks:

- Restoria_Document_Demo.ipynb – document restoration and enhancement
- Restoria_Ensemble_Demo.ipynb – combine multiple backends for higher quality
- Restoria_Benchmark.ipynb – quick benchmarks across backends/engines/metrics

Legacy:

- GFPGAN_Colab.ipynb – kept for backward compatibility; new work should use Restoria_Colab.ipynb

## Development Workflow

### Jupytext Integration

For development, notebooks can be paired with Python scripts using Jupytext:

```bash
# Install Jupytext
pip install jupytext

# Pair notebook with Python script
jupytext --set-formats ipynb,py notebook.ipynb

# Sync changes
jupytext --sync notebook.ipynb
```

Benefits:

- **Version control**: Python scripts are git-friendly
- **Code review**: Easier to review changes in .py format
- **Reproducibility**: Scripts can run in CI/CD pipelines
- **Collaboration**: Merge conflicts easier to resolve

### Local Development

Run notebooks locally with GFPGAN:

```bash
# Install with notebook dependencies
pip install -e ".[notebook,metrics,web]"

# Start Jupyter
jupyter lab

# Or use VS Code with Jupyter extension
code notebooks/
```

### API Integration

Notebooks can interact with the GFPGAN API server:

```python
import requests

# Start API server (in separate terminal)
# uvicorn services.api.main:app --port 8000

# Submit restoration job
response = requests.post("http://localhost:8000/restore",
    files={"file": open("input.jpg", "rb")},
    data={"version": "1.3", "upscale": 2}
)

result = response.json()
print(f"Restoration completed: {result['download_url']}")
```

## Best Practices

### Environment Setup

For consistent environments across platforms:

```bash
# Create isolated environment
python -m venv notebook_env
source notebook_env/bin/activate  # Windows: notebook_env\Scripts\activate

# Install with all notebook dependencies
pip install -e ".[notebook,metrics,web]"

# Install Jupyter extensions
jupyter labextension install @jupyterlab/toc
```

### Performance Optimization

#### GPU Acceleration

```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Set device for notebooks
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
```

#### Memory Management

```python
# For large image processing
torch.cuda.empty_cache()  # Clear GPU cache when needed

# Use smaller batch sizes for memory-constrained environments
batch_size = 1 if device == "cpu" else 4
```

### Testing and CI

Environment variable for automated testing:

```python
import os

# Check if running in CI environment
is_ci = os.getenv("CI", "false").lower() == "true"
nb_smoke = os.getenv("NB_CI_SMOKE", "false").lower() == "true"

if is_ci or nb_smoke:
    # Use smaller inputs and dry-run mode for testing
    input_size = (256, 256)
    dry_run = True
else:
    # Full processing for interactive use
    input_size = (512, 512)
    dry_run = False
```

## Notebook Structure

### Recommended Organization

```text
notebooks/
├── README.md                 # This file
├── Restoria_Colab.ipynb     # Main tutorial notebook
├── examples/                 # Example notebooks
│   ├── batch_processing.ipynb
│   ├── quality_metrics.ipynb
│   └── model_comparison.ipynb
├── scripts/                  # Jupytext paired scripts
│   ├── GFPGAN_Colab.py
│   └── examples/
└── outputs/                  # Generated outputs (gitignored)
    ├── restored_images/
    └── metrics_reports/
```

### Output Management

Notebook outputs are not stored in version control by design:

- **Text outputs**: Included for documentation
- **Image outputs**: Cleared before committing
- **Large files**: Stored in `outputs/` (gitignored)

```bash
# Clear notebook outputs before committing
jupyter nbconvert --clear-output --inplace *.ipynb

# Or use nbstripout for automatic clearing
pip install nbstripout
nbstripout --install
```

## Common Use Cases

### Quick Restoration

```python
from gfpgan.utils import restore_image

# Simple restoration
restored = restore_image(
    "input.jpg",
    output_path="output.jpg",
    version="1.3",
    upscale=2
)
```

### Batch Processing

```python
from pathlib import Path
from gfpgan import GFPGANer

# Initialize restorer
restorer = GFPGANer(model_path="GFPGANv1.3.pth", upscale=2)

# Process directory
input_dir = Path("input_images")
for img_path in input_dir.glob("*.jpg"):
    result = restorer.enhance(str(img_path))
    # Save and display results
```

### Quality Evaluation

```python
from gfpgan.metrics import calculate_metrics

# Compare restoration quality
metrics = calculate_metrics(
    original="original.jpg",
    restored="restored.jpg",
    metrics=["lpips", "dists", "arcface"]
)

print(f"Quality metrics: {metrics}")
```

## Getting Help

- **Documentation**: [User guides](../docs/guides/)
- **API Reference**: [API documentation](../docs/api/)
- **Issues**: [GitHub Issues](https://github.com/IAmJonoBo/Restoria/issues)
- **Discussions**: [GitHub Discussions](https://github.com/IAmJonoBo/Restoria/discussions)

