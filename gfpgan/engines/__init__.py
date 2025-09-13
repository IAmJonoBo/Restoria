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

# Placeholder for future engines (e.g., RestoreFormer++)
