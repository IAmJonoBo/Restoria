from typing import Any, Dict, Optional
import os
import tempfile

from .base import RestoreResult, Restorer


class GuidedRestorer(Restorer):
    """Reference-guided restoration (lightweight implementation).

        Behavior:
        - If a reference image path is provided (cfg["reference"]) and ArcFace is available,
            compute identity similarity between current input and reference. Use this to bias
            the restoration strength (weight) a bit toward identity preservation when similarity
            is low.
        - Runs GFPGAN underneath with the adjusted weight. If any dependency is missing, it
            gracefully falls back to the baseline GFPGAN path with the original cfg.
        """

    def __init__(self, device: str = "auto", bg_upsampler=None) -> None:
        self._device = device
        self._bg = bg_upsampler
        self._ref_path: Optional[str] = None
        self._inner = None

    def prepare(self, cfg: Dict[str, Any]) -> None:
        ref = cfg.get("reference")
        if isinstance(ref, str) and len(ref) > 0:
            self._ref_path = ref
        # Prepare inner baseline restorer lazily
        try:
            from .gfpgan import GFPGANRestorer
            self._inner = GFPGANRestorer(device=self._device, bg_upsampler=self._bg)
            # allow inner prepare later in restore with passed cfg
        except Exception:
            self._inner = None

    def restore(self, image: Any, cfg: Dict[str, Any]) -> RestoreResult:
        if self._ref_path is None or self._inner is None:
            self.prepare(cfg)
        # Default: pass-through if inner restorer unavailable
        if self._inner is None:
            return RestoreResult(
                input_path=cfg.get("input_path"),
                restored_path=None,
                restored_image=image,
                cropped_faces=[],
                restored_faces=[],
                metrics={
                    "guided": {
                        "reference": self._ref_path,
                        "arcface_cosine": None,
                        "adjusted_weight": None,
                    }
                },
            )

        # Compute optional identity similarity
        arc_cos = None
        adj_weight = None
        try:
            if self._ref_path and os.path.exists(self._ref_path):
                from ..metrics import ArcFaceIdentity  # lazy
                import cv2  # type: ignore

                arc = ArcFaceIdentity(no_download=bool(cfg.get("no_download", False)))
                if arc.available():
                    td = tempfile.mkdtemp()
                    a = os.path.join(td, "in.png")
                    b = os.path.join(td, "ref.png")
                    # Write current input and reference as BGR PNGs
                    cv2.imwrite(a, image)
                    ref_img = cv2.imread(self._ref_path)
                    if ref_img is not None:
                        cv2.imwrite(b, ref_img)
                        arc_cos = arc.cosine_from_paths(a, b)
        except Exception:
            arc_cos = None

        # Adjust weight subtly if similarity is low
        try:
            base_w = float(cfg.get("weight", 0.5))
        except Exception:
            base_w = 0.5
        if isinstance(arc_cos, float):
            if arc_cos < 0.2:
                adj_weight = max(0.3, base_w - 0.2)
            elif arc_cos < 0.35:
                adj_weight = max(0.35, base_w - 0.1)
            else:
                adj_weight = base_w
        # Build cfg for inner
        inner_cfg = dict(cfg)
        if adj_weight is not None:
            inner_cfg["weight"] = float(adj_weight)
        # Run inner restoration
        res = self._inner.restore(image, inner_cfg)
        # Attach guided metrics
        met = res.metrics or {}
        met.setdefault("guided", {})
        met["guided"].update(
            {
                "reference": self._ref_path,
                "arcface_cosine": arc_cos,
                "adjusted_weight": adj_weight,
            }
        )
        res.metrics = met
        return res
