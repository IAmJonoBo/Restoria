# Installation

## Quick install

The fastest way to get started:

```bash
pip install gfpgan
```

This installs the core GFPGAN package with basic dependencies.

## Platform-specific setup

### Windows

=== "NVIDIA GPU"
    ```bash
    # Install PyTorch with CUDA support first
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # Install GFPGAN
    pip install gfpgan
    ```

=== "DirectML (AMD/Intel)"
    ```bash
    # Install PyTorch with DirectML support
    pip install torch-directml

    # Install GFPGAN
    pip install gfpgan
    ```

=== "CPU only"
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install gfpgan
    ```

### macOS

=== "Apple Silicon (M1/M2/M3)"
    ```bash
    # Metal Performance Shaders (MPS) support included
    pip install gfpgan
    ```

=== "Intel Mac"
    ```bash
    # CPU-only installation
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install gfpgan
    ```

### Linux

=== "NVIDIA GPU"
    ```bash
    # Install PyTorch with CUDA support
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # Install GFPGAN
    pip install gfpgan
    ```

=== "AMD GPU (ROCm)"
    ```bash
    # Install PyTorch with ROCm support
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6

    # Install GFPGAN
    pip install gfpgan
    ```

=== "CPU only"
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install gfpgan
    ```

## Development installation

For contributing or using the latest features:

```bash
git clone https://github.com/IAmJonoBo/Restoria.git
cd Restoria
pip install -e ".[dev,metrics,web]"
```

### Optional extras

Install additional features as needed:

```bash
# Web interface and API
pip install -e ".[web]"

# Quality metrics
pip install -e ".[metrics]"

# Development tools
pip install -e ".[dev]"

# All extras
pip install -e ".[dev,metrics,web]"
```

## Verify installation

Test your installation:

```bash
# Check if GFPGAN is installed
gfpgan-infer --help

# Test with a simple command (dry run)
gfpgan-infer --input test.jpg --dry-run
```

## Troubleshooting

### Common issues

!!! error "CUDA out of memory"
    Try CPU mode or reduce batch size:
    ```bash
    gfpgan-infer --input photo.jpg --device cpu
    ```

!!! error "ModuleNotFoundError: No module named 'cv2'"
    Install OpenCV:
    ```bash
    pip install opencv-python
    ```

!!! error "No module named 'basicsr'"
    Install BasicSR:
    ```bash
    pip install basicsr
    ```

### Getting help

If you encounter issues:

1. Check our [troubleshooting guide](../troubleshooting.md)
2. Search [existing issues](https://github.com/IAmJonoBo/Restoria/issues)
3. Create a [new issue](https://github.com/IAmJonoBo/Restoria/issues/new) with:
   - Your platform (OS, GPU model)
   - Python version (`python --version`)
   - Error message and full traceback

### System requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.10+ |
| **RAM** | 4GB | 8GB+ |
| **GPU Memory** | N/A (CPU) | 4GB+ |
| **Storage** | 2GB | 5GB+ (for models) |

---

**Next:** [Restore your first photo →](../guides/restore-a-photo.md)
