from .codeformer_engine import CodeFormerEngine
from .gfpgan_engine import GFPGANEngine
from .registry import get_engine, register_engine

__all__ = [
    "register_engine",
    "get_engine",
    "GFPGANEngine",
    "CodeFormerEngine",
]

# Register default engines on import
register_engine("gfpgan", GFPGANEngine)
register_engine("codeformer", CodeFormerEngine)

# Placeholder for future engines (e.g., RestoreFormer++)
