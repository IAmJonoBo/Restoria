import argparse
import hashlib
import os
import sys
from urllib.request import urlretrieve

from .registry import load_model_registry


def load_registry():
    """Load model registry using the centralized registry system."""
    registry = load_model_registry()
    # Convert to format expected by CLI download tool
    converted = {}
    for name, meta in registry.items():
        converted[name] = {
            "url": meta.get("url", ""),
            "fname": meta.get("filename", f"{name}.pth")
        }
    return converted


def sha256sum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    def _progress(blocknum, blocksize, totalsize):  # pragma: no cover - cosmetic
        if totalsize <= 0:
            return
        read = blocknum * blocksize
        pct = min(100, int(read * 100 / totalsize))
        sys.stdout.write(f"\rDownloading {os.path.basename(dst)}: {pct}%")
        sys.stdout.flush()

    urlretrieve(url, dst, _progress)
    sys.stdout.write("\n")


def resolve_model_name(version: str) -> str:
    # Map version flag to model key used in inference
    mapping = {
        "1": "GFPGANv1",
        "1.2": "GFPGANCleanv1-NoCE-C2",
        "1.3": "GFPGANv1.3",
        "1.4": "GFPGANv1.4",
        "RestoreFormer": "RestoreFormer",
    }
    return mapping.get(version, version)


def main():
    parser = argparse.ArgumentParser(description="Download GFPGAN model weights to a local directory")
    parser.add_argument("--version", "-v", default=None, help="Model version or name (e.g., 1.4, 1.3, RestoreFormer)")
    parser.add_argument("--all", action="store_true", help="Download all known weights")
    parser.add_argument("--dir", default=os.path.join("gfpgan", "weights"), help="Destination directory")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    registry = load_registry()
    if args.list:
        print("Available models:")
        seen = set()
        for k, v in registry.items():
            if v["url"] in seen:
                continue
            seen.add(v["url"])
            print(f"- {k}: {v['url']}")
        return

    to_get = []
    if args.all:
        to_get = list({v["url"]: k for k, v in registry.items()}.values())
    elif args.version:
        model_key = resolve_model_name(args.version)
        if model_key not in registry:
            print(f"Unknown model '{args.version}'. Known: {', '.join(sorted(registry.keys()))}")
            sys.exit(1)
        to_get = [model_key]
    else:
        parser.error("Provide --version or --all or --list")

    for name in to_get:
        meta = registry[name]
        dst = os.path.join(args.dir, meta["fname"])
        if os.path.exists(dst):
            print(f"Exists: {dst} (sha256={sha256sum(dst)[:12]}...)")
            continue
        print(f"Fetching {name} → {dst}")
        download(meta["url"], dst)
        print(f"Done: {dst} (sha256={sha256sum(dst)[:12]}...)")


if __name__ == "__main__":  # pragma: no cover
    main()
