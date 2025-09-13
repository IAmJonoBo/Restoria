from typing import Any


class TensorRTEngine:
    """TensorRT engine (placeholder).

    Registers only if `tensorrt` is importable. Real implementation should
    build/load an engine and provide an `enhance` API like GFPGANer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        try:
            import tensorrt as trt  # type: ignore  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError("tensorrt not installed") from e
        raise NotImplementedError("TensorRTEngine not yet implemented")
