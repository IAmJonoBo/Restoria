import argparse
import glob
import os

# Defer heavy imports (cv2, numpy, torch, basicsr) until after --dry-run
imwrite = None  # will be set after delayed import


def _save_image(path, img, ext, jpg_quality=95, png_compress=3, webp_quality=90):
    """Write image with format-specific quality options using OpenCV.

    Falls back to defaults if OpenCV is unavailable.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import cv2

        params = []
        e = (ext or "").lower()
        if e in ("jpg", "jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
        elif e == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compress)]
        elif e == "webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, int(webp_quality)]
        cv2.imwrite(path, img, params)
    except Exception:
        # Fallback to basicsr.imwrite if available
        if imwrite is not None:
            imwrite(img, path)
        else:
            raise


def main():
    """Inference demo for GFPGAN (for users).

    Added UX flags:
    - --device: force cpu/cuda/auto
    - --dry-run: validate args and exit without loading models
    - --no-download: do not fetch remote weights if missing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="inputs/whole_imgs", help="Input image or folder. Default: inputs/whole_imgs"
    )
    parser.add_argument("-o", "--output", type=str, default="results", help="Output folder. Default: results")
    # we use version to select models, which is more user-friendly
    parser.add_argument(
        "-v", "--version", type=str, default="1.3", help="GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3"
    )
    parser.add_argument(
        "-s", "--upscale", type=int, default=2, help="The final upsampling scale of the image. Default: 2"
    )

    parser.add_argument(
        "--bg_upsampler",
        type=str,
        default="realesrgan",
        choices=["realesrgan", "none"],
        help="Background upsampler. Use 'none' to disable. Default: realesrgan",
    )
    parser.add_argument(
        "--bg_tile",
        type=int,
        default=400,
        help="Tile size for background sampler, 0 for no tile during testing. Default: 400",
    )
    parser.add_argument(
        "--bg_precision",
        type=str,
        default="auto",
        choices=["auto", "fp16", "fp32"],
        help="Precision for background upsampler: auto (cuda=fp16, cpu=fp32), fp16, or fp32.",
    )
    parser.add_argument("--suffix", type=str, default=None, help="Suffix of the restored faces")
    parser.add_argument("--only_center_face", action="store_true", help="Only restore the center face")
    parser.add_argument("--aligned", action="store_true", help="Input are aligned faces")
    parser.add_argument(
        "--ext",
        type=str,
        default="auto",
        help=(
            "Image extension. Options: auto | jpg | png | webp; "
            "auto means using the same extension as inputs. Default: auto"
        ),
    )
    parser.add_argument("-w", "--weight", type=float, default=0.5, help="Adjustable weights.")
    parser.add_argument(
        "--sweep-weight",
        type=str,
        default=None,
        help="Comma-separated list of weights to sweep (e.g., 0.3,0.5,0.7). Overrides --weight.",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use (default: auto)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse args and exit early.")
    parser.add_argument(
        "--no-download", action="store_true", help="Do not download remote weights when not present locally."
    )
    parser.add_argument("--model-path", type=str, default=None, help="Override model weights path (local file)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-cmp", action="store_true", help="Do not save side-by-side comparison tiles")
    # Output format quality controls
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--png-compress", type=int, default=3, help="PNG compression (0-9)")
    parser.add_argument("--webp-quality", type=int, default=90, help="WebP quality (1-100)")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images where outputs already exist")
    parser.add_argument(
        "--detector",
        type=str,
        default="retinaface_resnet50",
        choices=["retinaface_resnet50", "retinaface_mobile0.25", "scrfd"],
        help="Face detector backend (facexlib).",
    )
    parser.add_argument("--no-parse", action="store_true", help="Disable face parsing for blending.")
    parser.add_argument("--manifest", type=str, default=None, help="Write a JSON manifest of outputs to this path")
    parser.add_argument("--print-env", action="store_true", help="Print environment and versions then continue")
    parser.add_argument("--deterministic-cuda", action="store_true", help="Enable deterministic CuDNN (slower)")
    parser.add_argument("--eye-dist-threshold", type=int, default=5, help="Minimum eye distance (px) to detect a face")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith("/"):
        args.input = args.input[:-1]

    # Support files, directories, and glob patterns; filter to image-like files
    def _is_img(p):
        return os.path.splitext(p)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    if os.path.isfile(args.input):
        img_list = [args.input]
    elif os.path.isdir(args.input):
        img_list = sorted(x for x in glob.glob(os.path.join(args.input, "*")) if _is_img(x))
    else:
        img_list = sorted(x for x in glob.glob(args.input) if _is_img(x))

    os.makedirs(args.output, exist_ok=True)

    if args.dry_run:
        print("Dry run OK. Parsed arguments:")
        print(vars(args))
        return

    # Resolve device
    # Delayed heavy imports
    import torch
    from basicsr.utils import imwrite as _imwrite

    global imwrite  # set module-level alias
    imwrite = _imwrite

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device.replace("auto", "cpu")
    if args.verbose:
        print(f"Device: {device}")
    if args.print_env:
        try:
            import basicsr as _basicsr
            import facexlib as _facexlib
            import torchvision
        except Exception:
            torchvision = None  # type: ignore
            _basicsr = None  # type: ignore
            _facexlib = None  # type: ignore
        print("Env:")
        print(f"  torch={getattr(torch, '__version__', 'n/a')}")
        print(f"  torchvision={getattr(torchvision, '__version__', 'n/a') if torchvision else 'n/a'}")
        print(f"  basicsr={getattr(_basicsr, '__version__', 'n/a') if _basicsr else 'n/a'}")
        print(f"  facexlib={getattr(_facexlib, '__version__', 'n/a') if _facexlib else 'n/a'}")
        print(f"  cuda_available={torch.cuda.is_available()}")
    if args.deterministic_cuda and device == "cuda":
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------ set up background upsampler ------------------------
    if args.bg_upsampler == "realesrgan":
        if device == "cpu":  # CPU
            import warnings

            warnings.warn("RealESRGAN on CPU is slow; background upsampling disabled.")
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            if args.bg_precision == "auto":
                half = True
            else:
                half = args.bg_precision == "fp16"
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=half,
            )
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    if args.version == "1":
        arch = "original"
        channel_multiplier = 1
        model_name = "GFPGANv1"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth"
    elif args.version == "1.2":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANCleanv1-NoCE-C2"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth"
    elif args.version == "1.3":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANv1.3"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    elif args.version == "1.4":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANv1.4"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    elif args.version == "RestoreFormer":
        arch = "RestoreFormer"
        channel_multiplier = 2
        model_name = "RestoreFormer"
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    else:
        raise ValueError(f"Wrong model version {args.version}.")

    # determine model paths
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join("experiments/pretrained_models", model_name + ".pth")
        if not os.path.isfile(model_path):
            model_path = os.path.join("gfpgan/weights", model_name + ".pth")
        if not os.path.isfile(model_path):
            if args.no_download:
                raise FileNotFoundError(f"Model weights {model_name}.pth not found locally and --no-download is set.")
            # download pre-trained models from url
            model_path = url

    # Delay heavy import until after dry-run
    from gfpgan import GFPGANer

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler,
        device=torch.device(device),
        det_model=args.detector,
        use_parse=not args.no_parse,
    )

    # Optional seeding
    if args.seed is not None:
        try:
            import random

            random.seed(args.seed)
        except Exception:
            pass
        try:
            import numpy as _np

            _np.random.seed(args.seed)
        except Exception:
            pass
        try:
            torch.manual_seed(args.seed)
        except Exception:
            pass

    # ------------------------ restore ------------------------
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover

        def _tqdm_passthrough(x):  # E731: use def instead of lambda
            return x

        tqdm = _tqdm_passthrough  # type: ignore

    # Optional limit
    if args.max_images is not None:
        img_list = img_list[: max(0, args.max_images)]

    results = []
    for img_path in tqdm(img_list):
        # read image
        img_name = os.path.basename(img_path)
        if args.verbose:
            print(f"Processing {img_name} ...")
        basename, ext = os.path.splitext(img_name)
        import cv2
        import numpy as np

        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        # Determine weights to run
        if args.sweep_weight:
            try:
                weights = [float(x.strip()) for x in args.sweep_weight.split(",") if x.strip()]
            except Exception:
                raise ValueError("Invalid --sweep-weight; expected comma-separated floats")
        else:
            weights = [args.weight]

        face_paths = []
        crop_paths = []
        save_restore_paths = []

        for w in weights:
            # If skipping existing outputs, check the expected restored image path
            exp_suffix = args.suffix
            if exp_suffix is None and len(weights) > 1:
                exp_suffix = f"w{w:.2f}"

            # Run enhancement
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                input_img,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True,
                weight=w,
            )

            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                # save cropped face
                save_crop_path = os.path.join(args.output, "cropped_faces", f"{basename}_{idx:02d}.png")
                _save_image(
                    save_crop_path,
                    cropped_face,
                    ext="png",
                    jpg_quality=args.jpg_quality,
                    png_compress=args.png_compress,
                    webp_quality=args.webp_quality,
                )
                crop_paths.append(save_crop_path)
                # save restored face
                tag = exp_suffix
                if tag is not None:
                    save_face_name = f"{basename}_{idx:02d}_{tag}.png"
                else:
                    save_face_name = f"{basename}_{idx:02d}.png"
                save_restore_path = os.path.join(args.output, "restored_faces", save_face_name)
                _save_image(
                    save_restore_path,
                    restored_face,
                    ext="png",
                    jpg_quality=args.jpg_quality,
                    png_compress=args.png_compress,
                    webp_quality=args.webp_quality,
                )
                face_paths.append(save_restore_path)
                # save comparison image
                if not args.no_cmp:
                    cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                    _save_image(
                        os.path.join(args.output, "cmp", f"{basename}_{idx:02d}.png"),
                        cmp_img,
                        ext="png",
                        jpg_quality=args.jpg_quality,
                        png_compress=args.png_compress,
                        webp_quality=args.webp_quality,
                    )

            # save restored img
            if restored_img is not None:
                if args.ext == "auto":
                    extension = ext[1:]
                else:
                    extension = args.ext

                tag = exp_suffix
                if tag is not None:
                    out_name = f"{basename}_{tag}.{extension}"
                else:
                    out_name = f"{basename}.{extension}"
                out_path = os.path.join(args.output, "restored_imgs", out_name)
                if args.skip_existing and os.path.exists(out_path):
                    if args.verbose:
                        print(f"Skip existing: {out_path}")
                else:
                    _save_image(
                        out_path,
                        restored_img,
                        ext=extension,
                        jpg_quality=args.jpg_quality,
                        png_compress=args.png_compress,
                        webp_quality=args.webp_quality,
                    )
                save_restore_paths.append(out_path)
            else:
                save_restore_paths.append(None)

        results.append(
            {
                "input": img_path,
                "restored_imgs": save_restore_paths,
                "restored_faces": face_paths,
                "cropped_faces": crop_paths,
                "weights": weights,
            }
        )

    if args.manifest:
        import json

        with open(args.manifest, "w") as f:
            json.dump({"results": results}, f, indent=2)
        if args.verbose:
            print(f"Wrote manifest: {args.manifest}")

    print(f"Results are in the [{args.output}] folder.")


if __name__ == "__main__":
    main()
