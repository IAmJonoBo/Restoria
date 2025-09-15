"""
Central model registry system for GFPGAN.

This module provides a unified way to load and access model information
across all GFPGAN components.
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_model_registry() -> Dict[str, Dict[str, Any]]:
    """
    Load model registry from models/registry.yml with fallback to built-in registry.

    Returns:
        Dictionary mapping model names and aliases to their metadata including URLs.
    """
    # Try to load from models/registry.yml first
    registry_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "registry.yml"
    )

    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Build alias map
            registry = {}
            for name, meta in data.items():
                meta = meta or {}
                # Extract filename from URL or generate default
                url = meta.get("url", "")
                filename = _extract_filename_from_url(url) or f"{name}.pth"

                entry = {
                    "url": url,
                    "filename": filename,
                    "description": meta.get("description", ""),
                    "aliases": meta.get("aliases", [])
                }

                # Add main entry
                registry[name] = entry

                # Add alias entries
                for alias in meta.get("aliases", []):
                    registry[alias] = entry

            return registry

        except Exception as e:
            print(f"Warning: Could not load registry from {registry_path}: {e}")

    # Fallback to built-in registry with updated URLs
    return _get_fallback_registry()


def _extract_filename_from_url(url: str) -> Optional[str]:
    """Extract filename from a URL."""
    if not url:
        return None

    # Handle URL-encoded filenames
    import urllib.parse
    decoded_url = urllib.parse.unquote(url)

    # Extract the last part of the path
    parts = decoded_url.split('/')
    if parts:
        filename = parts[-1]
        # Remove query parameters if any
        if '?' in filename:
            filename = filename.split('?')[0]
        return filename

    return None


def _get_fallback_registry() -> Dict[str, Dict[str, Any]]:
    """Fallback registry with verified working URLs."""
    base_registry = {
        "GFPGANv1": {
            "url": "https://huggingface.co/TencentARC/GFPGANv1/resolve/main/GFPGANv1.pth",
            "filename": "GFPGANv1.pth",
            "description": "Original GFPGAN v1 model with colorization",
            "aliases": ["1", "v1"]
        },
        "GFPGANCleanv1-NoCE-C2": {
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth",
            "filename": "GFPGANCleanv1-NoCE-C2.pth",
            "description": "Clean version without colorization, no CUDA extensions required",
            "aliases": ["1.2", "v1.2", "clean"]
        },
        "GFPGANv1.3": {
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            "filename": "GFPGANv1.3.pth",
            "description": "Latest stable GFPGAN - more natural restoration, better on low/high quality inputs",
            "aliases": ["1.3", "v1.3"]
        },
        "RestoreFormer": {
            "url": "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer.ckpt",
            "filename": "RestoreFormer.ckpt",
            "description": "RestoreFormer original model from RestoreFormer++ repository",
            "aliases": ["restoreformer", "rf"]
        },
        "RestoreFormerPlusPlus": {
            "url": "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer%2B%2B.ckpt",
            "filename": "RestoreFormer++.ckpt",
            "description": "Latest RestoreFormer++ model (TPAMI 2023) - improved version of RestoreFormer",
            "aliases": ["restoreformer++", "rfpp", "rf++"]
        },
        "CodeFormer": {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            "filename": "codeformer.pth",
            "description": "CodeFormer main model - robust face restoration for old photos and AI-generated faces",
            "aliases": ["codeformer", "cf"]
        },
        "CodeFormerColorization": {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_colorization.pth",
            "filename": "codeformer_colorization.pth",
            "description": "CodeFormer model specialized for colorization tasks",
            "aliases": ["codeformer_colorization", "cf_color"]
        },
        "CodeFormerInpainting": {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth",
            "filename": "codeformer_inpainting.pth",
            "description": "CodeFormer model specialized for inpainting tasks",
            "aliases": ["codeformer_inpainting", "cf_inpaint"]
        },
        "RealESRGAN_x2plus": {
            "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
            "filename": "RealESRGAN_x2plus.pth",
            "description": "RealESRGAN x2 model for background upsampling",
            "aliases": ["realesrgan", "real_esrgan"]
        }
    }

    # Build final registry with aliases
    registry = {}
    for name, meta in base_registry.items():
        registry[name] = meta
        for alias in meta.get("aliases", []):
            registry[alias] = meta

    return registry


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get model information by name or alias.

    Args:
        model_name: Model name or alias

    Returns:
        Model metadata dictionary or None if not found
    """
    registry = load_model_registry()
    return registry.get(model_name)


def list_available_models() -> Dict[str, str]:
    """
    List all available models with their descriptions.

    Returns:
        Dictionary mapping model names to descriptions
    """
    registry = load_model_registry()
    models = {}

    # Only include main model names (not aliases)
    seen_urls = set()
    for name, meta in registry.items():
        url = meta.get("url", "")
        if url and url not in seen_urls:
            models[name] = meta.get("description", "")
            seen_urls.add(url)

    return models


def get_model_architecture_info(model_name: str) -> Dict[str, Any]:
    """
    Get architecture-specific information for a model.

    Args:
        model_name: Model name or alias

    Returns:
        Dictionary with architecture information
    """
    model_info = get_model_info(model_name)
    if not model_info:
        return {}

    # Map model names to their architecture configurations
    arch_mapping = {
        "GFPGANv1": {"arch": "original", "channel_multiplier": 1},
        "GFPGANCleanv1-NoCE-C2": {"arch": "clean", "channel_multiplier": 2},
        "GFPGANv1.3": {"arch": "clean", "channel_multiplier": 2},
        "RestoreFormer": {"arch": "RestoreFormer", "channel_multiplier": 2},
        "RestoreFormerPlusPlus": {"arch": "RestoreFormer", "channel_multiplier": 2},
        "CodeFormer": {"arch": "codeformer", "channel_multiplier": 2},
        "CodeFormerColorization": {"arch": "codeformer", "channel_multiplier": 2},
        "CodeFormerInpainting": {"arch": "codeformer", "channel_multiplier": 2},
    }

    # Find the main model name (not alias)
    registry = load_model_registry()
    main_name = None
    for name, meta in registry.items():
        if meta.get("url") == model_info.get("url") and name in arch_mapping:
            main_name = name
            break

    if main_name:
        return arch_mapping[main_name]

    # Default fallback
    return {"arch": "clean", "channel_multiplier": 2}