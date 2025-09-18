def test_list_backends_no_plugins():
    # Import inside test to avoid import-time side effects
    from restoria.core.registry import list_backends  # type: ignore

    backs = list_backends()
    # Must include built-ins, even if plugins missing
    assert isinstance(backs, dict)
    assert "gfpgan" in backs
    assert "gfpgan-ort" in backs
    # Function should not raise and should return availability metadata
    for _name, info in backs.items():
        assert isinstance(info, dict)
        assert isinstance(info.get("available"), bool)
        assert "metadata" in info
