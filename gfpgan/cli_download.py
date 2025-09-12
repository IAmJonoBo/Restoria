import argparse
import hashlib
import os
import sys
from urllib.request import urlretrieve

REGISTRY = {
    "GFPGANv1": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth",
        "fname": "GFPGANv1.pth",
    },
    "GFPGANCleanv1-NoCE-C2": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth",
        "fname": "GFPGANCleanv1-NoCE-C2.pth",
    },
    "GFPGANv1.3": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        "fname": "GFPGANv1.3.pth",
    },
    "GFPGANv1.4": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "fname": "GFPGANv1.4.pth",
    },
    "RestoreFormer": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
        "fname": "RestoreFormer.pth",
    },
}


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

    if args.list:
        print("Available models:")
        for k, v in REGISTRY.items():
            print(f"- {k}: {v['url']}")
        return

    to_get = []
    if args.all:
        to_get = list(REGISTRY.keys())
    elif args.version:
        model_key = resolve_model_name(args.version)
        if model_key not in REGISTRY:
            print(f"Unknown model '{args.version}'. Known: {', '.join(REGISTRY.keys())}")
            sys.exit(1)
        to_get = [model_key]
    else:
        parser.error("Provide --version or --all or --list")

    for name in to_get:
        meta = REGISTRY[name]
        dst = os.path.join(args.dir, meta["fname"])
        if os.path.exists(dst):
            print(f"Exists: {dst} (sha256={sha256sum(dst)[:12]}...)")
            continue
        print(f"Fetching {name} → {dst}")
        download(meta["url"], dst)
        print(f"Done: {dst} (sha256={sha256sum(dst)[:12]}...)")


if __name__ == "__main__":  # pragma: no cover
    main()
