from __future__ import annotations

from typing import Optional


class LPIPSMetric:
    def __init__(self) -> None:
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net="alex")
        except Exception:
            self.model = None

    def available(self) -> bool:
        return self.model is not None

    def distance_from_paths(self, a_path: str, b_path: str) -> Optional[float]:
        if self.model is None:
            return None
        try:
            import cv2  # type: ignore
            import torch  # type: ignore

            a = cv2.imread(a_path)
            b = cv2.imread(b_path)
            if a is None or b is None:
                return None
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a = cv2.resize(a, (w, h))
            b = cv2.resize(b, (w, h))
            at = torch.from_numpy(((a[:, :, ::-1].astype("float32") / 255.0) * 2 - 1)).permute(2, 0, 1).unsqueeze(0)
            bt = torch.from_numpy(((b[:, :, ::-1].astype("float32") / 255.0) * 2 - 1)).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                return float(self.model(at, bt).item())
        except Exception:
            return None

