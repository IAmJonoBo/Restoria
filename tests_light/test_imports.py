def test_imports():
    # Light import/sanity tests (CPU-friendly)
    import importlib

    assert importlib.import_module("gfpgan") is not None
    assert importlib.import_module("gfpgan.utils") is not None
    assert importlib.import_module("gfpgan.archs") is not None
    assert importlib.import_module("gfpgan.models") is not None
