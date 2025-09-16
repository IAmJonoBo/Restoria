from .registry import get_engine, register_engine

__all__ = [
    "register_engine",
    "get_engine",
]

# Register default engines on import with lazy imports to avoid heavy deps at import time
try:
    from .gfpgan_engine import GFPGANEngine  # type: ignore

    register_engine("gfpgan", GFPGANEngine)
except Exception:  # pragma: no cover
    pass

try:
    from .codeformer_engine import CodeFormerEngine  # type: ignore

    register_engine("codeformer", CodeFormerEngine)
except Exception:  # pragma: no cover
    pass

try:
    from .restoreformer_engine import RestoreFormerEngine  # type: ignore

    register_engine("restoreformer", RestoreFormerEngine)
    register_engine("restoreformerpp", RestoreFormerEngine)
except Exception:  # pragma: no cover
    pass

# Optional accelerators: register only if importable
try:  # pragma: no cover
    from .onnx_engine import ONNXEngine  # type: ignore

    register_engine("onnx", ONNXEngine)
except Exception:
    pass
try:  # pragma: no cover
    from .tensorrt_engine import TensorRTEngine  # type: ignore

    register_engine("tensorrt", TensorRTEngine)
except Exception:
    pass

# Placeholder for future engines (e.g., RestoreFormer++)
