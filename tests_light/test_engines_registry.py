def test_engines_registry_has_default_entries():
    from gfpgan.engines import get_engine

    assert callable(get_engine("gfpgan"))
    assert callable(get_engine("codeformer"))
    assert callable(get_engine("restoreformer"))
    assert callable(get_engine("restoreformerpp"))
