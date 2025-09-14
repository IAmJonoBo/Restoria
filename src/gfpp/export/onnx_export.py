from __future__ import annotations

from typing import Optional


def export_gfpgan_onnx(version: str = "1.4", model_path: Optional[str] = None, out_path: str = "gfpgan.onnx") -> None:
    """Export a GFPGAN generator to ONNX (placeholder).

    This function outlines the intended export path but does not perform a real export
    to avoid heavy dependencies and large artifacts in this repo.

    Steps to implement:
    1) Resolve weights using gfpgan.weights.resolve_model_weight (or accept a local path)
    2) Initialize GFPGANRestorer and fetch the underlying generator module (restorer.gfpgan)
    3) Prepare a 512x512 normalized dummy input tensor and call torch.onnx.export with dynamic axes
    4) Save to out_path and verify with onnxruntime.InferenceSession
    """
    raise NotImplementedError(
        "ONNX export placeholder. To export: initialize restorer.gfpgan and call torch.onnx.export; "
        "then verify with onnxruntime."
    )
