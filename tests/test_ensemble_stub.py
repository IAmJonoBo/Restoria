from __future__ import annotations


def test_ensemble_restorer_instantiation():
    from src.gfpp.restorers.ensemble import EnsembleRestorer

    e = EnsembleRestorer(device="cpu", bg_upsampler=None)
    cfg = {"ensemble_backends": "gfpgan,codeformer", "ensemble_weights": "0.5,0.5"}
    # Should not raise
    e.prepare(cfg)
