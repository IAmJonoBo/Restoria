from __future__ import annotations


def test_adv_quality_wrappers_available_gated():
    from src.gfpp.metrics.adv_quality import MANIQAWrapper, CONTRIQUEWrapper, advanced_scores

    m = MANIQAWrapper()
    c = CONTRIQUEWrapper()
    # available() may be True depending on environment; score() must not crash
    assert m.score("/nonexistent.png") is None
    assert c.score("/nonexistent.png") is None
    s = advanced_scores("/nonexistent.png")
    assert isinstance(s, dict)


def test_document_preset_helper():
    from src.gfpp.presets import apply_preset

    cfg = {"weight": 0.5}
    out = apply_preset("document", cfg)
    assert out["weight"] <= 0.5
