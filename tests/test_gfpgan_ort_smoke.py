import os
import sys
import pytest

pytestmark = pytest.mark.heavy


def test_gfpgan_ort_smoke(tmp_path):
    # Ensure src on path
    sys.path.insert(0, 'src')
    try:
        __import__("onnxruntime")
    except Exception:
        pytest.skip("onnxruntime not available", allow_module_level=False)

    try:
        from gfpp.restorers.gfpgan_ort import ORTGFPGANRestorer  # type: ignore
    except Exception:
        pytest.skip("ORT restorer import failed (optional)", allow_module_level=False)
    # Fallback path depends on the gfpgan package being installed
    try:
        __import__("gfpgan")
    except Exception:
        pytest.skip("gfpgan package not available for fallback path", allow_module_level=False)

    # Create a dummy ONNX file; session should fail to init and fallback to torch
    dummy_model = os.path.join(tmp_path, 'gfpgan.onnx')
    with open(dummy_model, 'wb') as f:
        f.write(b"00")

    rest = ORTGFPGANRestorer(device='cpu', bg_upsampler=None)
    cfg = {
        'input_path': None,
        'upscale': 1,
        'use_parse': False,
        'detector': 'retinaface_resnet50',
        'model_path_onnx': dummy_model,
        'compile': 'none',
    }

    # Minimal 1x1 image
    import numpy as np  # type: ignore
    img = (np.zeros((1, 1, 3), dtype='uint8'))

    res = rest.restore(img, cfg)
    assert res is not None
    assert isinstance(res.metrics, dict)
    # backend should be torch-fallback or onnxruntime+torch-fallback due to invalid graph
    assert 'backend' in res.metrics
    assert 'onnxruntime' in str(res.metrics.get('backend')) or 'torch' in str(res.metrics.get('backend'))
