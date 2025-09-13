from .registry import register_engine, get_engine
from .gfpgan_engine import GFPGANEngine
from .codeformer_engine import CodeFormerEngine

# Register default engines on import
register_engine("gfpgan", GFPGANEngine)
register_engine("codeformer", CodeFormerEngine)

# Placeholder for future engines (e.g., RestoreFormer++)
