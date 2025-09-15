from typing import Any


class ONNXEngine:
    """ONNX Runtime engine (placeholder).

    Registers only if `onnxruntime` is importable. Real implementation should
    load a compatible model and provide an `enhance` API like GFPGANer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        try:
            import onnxruntime as ort  # type: ignore  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError("onnxruntime not installed") from e
        raise NotImplementedError("ONNXEngine not yet implemented")
