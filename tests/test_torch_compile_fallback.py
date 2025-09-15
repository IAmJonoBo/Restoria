import sys


def test_compile_module_mode_none_returns_original():
    sys.path.insert(0, 'src')
    from gfpp.engines.torch_compile import compile_module  # type: ignore

    dummy = object()
    out = compile_module(dummy, mode="none")
    assert out is dummy


def test_compile_module_import_failure_returns_original(monkeypatch):
    sys.path.insert(0, 'src')
    from gfpp.engines import torch_compile as tc  # type: ignore

    # Ensure importing torch inside compile_module will fail by placing a sentinel that raises on attribute access
    class _TorchSentinel:
        def __getattr__(self, name):  # any access triggers failure like compile_module expects
            raise RuntimeError("sentinel access")

    monkeypatch.setitem(sys.modules, 'torch', _TorchSentinel())

    dummy = object()
    out = tc.compile_module(dummy, mode="default")
    assert out is dummy
