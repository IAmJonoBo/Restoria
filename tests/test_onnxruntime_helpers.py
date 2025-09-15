import os
import sys


def test_available_eps_returns_list_or_empty():
    sys.path.insert(0, 'src')
    from gfpp.engines.onnxruntime import available_eps  # type: ignore

    eps = available_eps()
    assert isinstance(eps, list)


def test_create_session_missing_onnxruntime_returns_none(tmp_path):
    sys.path.insert(0, 'src')
    from gfpp.engines.onnxruntime import create_session  # type: ignore

    # Use a dummy model path; in absence of ORT this must return None
    dummy_model = os.path.join(tmp_path, 'model.onnx')
    with open(dummy_model, 'wb') as f:
        f.write(b"00")
    sess = create_session(dummy_model, providers=["CPUExecutionProvider"])  # explicit provider
    # Either ORT is present and may fail on invalid model returning None, or ORT missing returns None
    assert sess is None
