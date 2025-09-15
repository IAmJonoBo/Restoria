from __future__ import annotations

import os
from typing import Optional


def try_load_arcface(no_download: bool = False):
    try:
        import torch  # type: ignore
        from basicsr.utils.download_util import load_file_from_url  # type: ignore

        from gfpgan.archs.arcface_arch import ResNetArcFace  # type: ignore

        arc_path = os.environ.get(
            "ARCFACE_WEIGHTS",
            os.path.join(os.path.dirname(__file__), "weights", "arcface_resnet18.pth"),
        )
        if not os.path.isfile(arc_path):
            if no_download:
                return None
            arc_path = load_file_from_url(
                url="https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth",
                model_dir=os.path.join(os.path.dirname(__file__), "weights"),
                file_name="arcface_resnet18.pth",
                progress=True,
            )
        model = ResNetArcFace(block="IRBlock", layers=(2, 2, 2, 2), use_se=False)
        model.load_state_dict(torch.load(arc_path, map_location="cpu", weights_only=True))
        model.eval()
        return model
    except Exception:
        return None


def identity_cosine_from_paths(a_path: str, b_path: str, id_model) -> Optional[float]:
    try:
        import cv2  # type: ignore
        import torch  # type: ignore

        a = cv2.imread(a_path)
        b = cv2.imread(b_path)
        if a is None or b is None:
            return None

        def _prep(x):
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (112, 112))
            x = (x.astype("float32") / 255.0 - 0.5) / 0.5
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
            return x

        with torch.no_grad():
            f1 = id_model(_prep(a)).flatten().numpy()
            f2 = id_model(_prep(b)).flatten().numpy()
        return float((f1 @ f2) / ((float((f1**2).sum()) ** 0.5 + 1e-8) * (float((f2**2).sum()) ** 0.5 + 1e-8)))
    except Exception:
        return None


def try_lpips_model():
    try:
        import lpips  # type: ignore

        return lpips.LPIPS(net="alex")
    except Exception:
        return None


def lpips_from_paths(a_path: str, b_path: str, model) -> Optional[float]:
    try:
        import cv2  # type: ignore
        import torch  # type: ignore

        a = cv2.imread(a_path)
        b = cv2.imread(b_path)
        if a is None or b is None:
            return None
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        # Ensure LPIPS input spatial dims are not too small for AlexNet trunk.
        # AlexNet downsampling requires at least ~32-64px. Use 64px floor.
        floor = 64
        if h < floor or w < floor:
            h = max(h, floor)
            w = max(w, floor)
        a = cv2.resize(a, (w, h))
        b = cv2.resize(b, (w, h))

        def _to_t(x):
            x = x[:, :, ::-1]
            x = (x.astype("float32") / 255.0) * 2 - 1
            return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            return float(model(_to_t(a), _to_t(b)).item())
    except Exception:
        return None
