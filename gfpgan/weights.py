import hashlib
import os
from typing import Optional, Tuple

from .registry import load_model_registry, get_model_info

# Default fallback location inside the package; do NOT bake env here to allow
# tests to monkeypatch GFPGAN_WEIGHTS_DIR after import time.
DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


def get_model_registry():
    """Get the current model registry."""
    return load_model_registry()


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
    # Respect GFPGAN_WEIGHTS_DIR at call time (not only import time)
    weights_dir = root or os.environ.get("GFPGAN_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR)
    _ensure_dir(weights_dir)
    # Prefer a file named after the model_name if present (helps alias/offline compatibility)
    alias_path = os.path.join(weights_dir, f"{model_name}.pth")
    if os.path.isfile(alias_path):
        return alias_path, _sha256(alias_path)

    spec = get_model_info(model_name)
    if spec is None:
        # Ensure keys are converted to strings in case YAML produced non-string keys
        available_models = ", ".join(str(k) for k in load_model_registry().keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")

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

            # Use specified revision, environment variable, or secure default
            revision = spec.get("revision") or os.environ.get("GFPGAN_HF_REVISION") or "main"

            # Ensure revision is explicitly set for security
            if not spec.get("revision") and not os.environ.get("GFPGAN_HF_REVISION"):
                # Log warning that we're using a default revision
                import warnings

                warnings.warn(
                    f"Model {model_name}: Using default 'main' branch for HuggingFace download. "
                    f"Consider pinning to a specific revision in model spec or GFPGAN_HF_REVISION env var.",
                    UserWarning,
                    stacklevel=2,
                )

            path = hf_hub_download(
                repo_id=hf_repo,
                filename=spec["filename"],
                subfolder=spec.get("subfolder"),
                revision=revision,  # Always explicitly specify revision
                cache_dir=None,  # Use default cache
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

        path = load_file_from_url(url=spec["url"], model_dir=weights_dir, progress=True, file_name=spec["filename"])
        return path, _sha256(path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to download weight from {spec['url']}: {e}") from e
