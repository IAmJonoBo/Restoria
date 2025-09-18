from __future__ import annotations

from src.gfpp.cli import _instantiate_restorer


class _Args:
    def __init__(self, device="cpu", compile="none", allow_noncommercial=False):
        self.device = device
        self.compile = compile
        self.allow_noncommercial = allow_noncommercial


def test_codeformer_gated_by_default(monkeypatch):
    # Ensure env not set
    monkeypatch.delenv("RESTORIA_ALLOW_NONCOMMERCIAL", raising=False)
    _, name = _instantiate_restorer("codeformer", _Args(), bg=None)
    # Falls back to GFPGAN by default
    assert name == "gfpgan"


def test_codeformer_allowed_with_flag(monkeypatch):
    monkeypatch.delenv("RESTORIA_ALLOW_NONCOMMERCIAL", raising=False)
    _, name = _instantiate_restorer("codeformer", _Args(allow_noncommercial=True), bg=None)
    # Will attempt to use CodeFormer; may still be unavailable at runtime, but the gate opens here
    assert name == "codeformer"


def test_codeformer_allowed_with_env(monkeypatch):
    monkeypatch.setenv("RESTORIA_ALLOW_NONCOMMERCIAL", "1")
    _, name = _instantiate_restorer("codeformer", _Args(), bg=None)
    assert name == "codeformer"
