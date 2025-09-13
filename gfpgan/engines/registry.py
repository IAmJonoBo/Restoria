from typing import Any, Callable, Dict

_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_engine(name: str, cls: Callable[..., Any]) -> None:
    _REGISTRY[name] = cls


def get_engine(name: str) -> Callable[..., Any]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown engine '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
