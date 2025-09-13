import hashlib
import os
from typing import Optional, Tuple


DEFAULT_WEIGHTS_DIR = os.environ.get(
    "GFPGAN_WEIGHTS_DIR", os.path.join(os.path.dirname(__file__), "weights")
)


# Minimal built-in registry for core models. Hugging Face repos may not always
# host these weights; provide URLs as fallback. Users can override via envs.
MODEL_REGISTRY = {
    "GFPGANv1": {
        "filename": "GFPGANv1.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth",
        "hf_repo": os.environ.get("GFPGAN_HF_REPO", ""),
    },
    "GFPGANCleanv1-NoCE-C2": {
        "filename": "GFPGANCleanv1-NoCE-C2.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth",
        "hf_repo": os.environ.get("GFPGAN_HF_REPO", ""),
    },
    "GFPGANv1.3": {
        "filename": "GFPGANv1.3.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        "hf_repo": os.environ.get("GFPGAN_HF_REPO", ""),
    },
    "GFPGANv1.4": {
        "filename": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "hf_repo": os.environ.get("GFPGAN_HF_REPO", ""),
    },
    "RestoreFormer": {
        "filename": "RestoreFormer.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
        "hf_repo": os.environ.get("GFPGAN_HF_REPO", ""),
    },
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def resolve_model_weight(
    model_name: str,
    *,
    no_download: bool = False,
    root: Optional[str] = None,
    prefer: str = "auto",  # auto|hf|url
) -> Tuple[str, Optional[str]]:
    """Resolve a local path for the given model weight.

    Returns (path, sha256) if found/resolved; raises on failure when offline.
    Honors envs:
      - GFPGAN_WEIGHTS_DIR: destination directory for weights
      - GFPGAN_HF_REPO: Hugging Face repo to use (if set)
      - HF_HUB_OFFLINE: if '1', forces local cache-only
    """
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown model '{model_name}'")

    weights_dir = root or DEFAULT_WEIGHTS_DIR
    _ensure_dir(weights_dir)
    local_path = os.path.join(weights_dir, spec["filename"])
    if os.path.isfile(local_path):
        return local_path, _sha256(local_path)

    # Offline guard
    offline = no_download or os.environ.get("HF_HUB_OFFLINE") == "1"
    prefer = prefer if prefer in {"auto", "hf", "url"} else "auto"

    # Try huggingface hub if requested or auto with repo configured
    hf_repo = spec.get("hf_repo") or os.environ.get("GFPGAN_HF_REPO")
    if (prefer in {"auto", "hf"}) and hf_repo:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            path = hf_hub_download(
                repo_id=hf_repo,
                filename=spec["filename"],
                local_files_only=offline,
            )
            # Optionally copy/symlink to weights_dir; return cached path directly
            return path, _sha256(path)
        except Exception:
            if prefer == "hf":
                raise
            # fall through to URL

    # URL fallback via basicsr download util
    if offline:
        raise FileNotFoundError(
            f"Offline and weight not found locally: {local_path}. Provide the file or unset --no-download."
        )
    try:
        from basicsr.utils.download_util import load_file_from_url  # type: ignore

        path = load_file_from_url(
            url=spec["url"], model_dir=weights_dir, progress=True, file_name=spec["filename"]
        )
        return path, _sha256(path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to download weight from {spec['url']}: {e}")

