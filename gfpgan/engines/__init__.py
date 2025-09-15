from .codeformer_engine import CodeFormerEngine
from .gfpgan_engine import GFPGANEngine
from .registry import get_engine, register_engine
from .restoreformer_engine import RestoreFormerEngine

__all__ = [
    "register_engine",
    "get_engine",
    "GFPGANEngine",
    "CodeFormerEngine",
    "RestoreFormerEngine",
]

# Register default engines on import
register_engine("gfpgan", GFPGANEngine)
register_engine("codeformer", CodeFormerEngine)
register_engine("restoreformer", RestoreFormerEngine)
register_engine("restoreformerpp", RestoreFormerEngine)

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
