from src.gfpp.presets import apply_preset


def test_document_preset_weight_and_parse():
    cfg = {"weight": 0.6, "use_parse": False}
    out = apply_preset("document", cfg)
    # Non-destructive defaults: preset should not override explicit weight
    from pytest import approx
    assert out["weight"] == approx(0.6)
    # But should set other defaults if absent
    out2 = apply_preset("document", {})
    assert out2["weight"] == approx(0.3)
    assert out2["use_parse"] is True
