import argparse
import glob
import os

import cv2
import numpy as np
import torch
from basicsr.utils import imwrite


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
        "--bg_upsampler", type=str, default="realesrgan", help="background upsampler. Default: realesrgan"
    )
    parser.add_argument(
        "--bg_tile",
        type=int,
        default=400,
        help="Tile size for background sampler, 0 for no tile during testing. Default: 400",
    )
    parser.add_argument("--suffix", type=str, default=None, help="Suffix of the restored faces")
    parser.add_argument("--only_center_face", action="store_true", help="Only restore the center face")
    parser.add_argument("--aligned", action="store_true", help="Input are aligned faces")
    parser.add_argument(
        "--ext",
        type=str,
        default="auto",
        help="Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto",
    )
    parser.add_argument("-w", "--weight", type=float, default=0.5, help="Adjustable weights.")
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
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith("/"):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, "*")))

    os.makedirs(args.output, exist_ok=True)

    if args.dry_run:
        print("Dry run OK. Parsed arguments:")
        print(vars(args))
        return

    # Resolve device
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device.replace("auto", "cpu")

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
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True,
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

    for img_path in tqdm(img_list):
        # read image
        img_name = os.path.basename(img_path)
        print(f"Processing {img_name} ...")
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=args.aligned,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.weight,
        )

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save cropped face
            save_crop_path = os.path.join(args.output, "cropped_faces", f"{basename}_{idx:02d}.png")
            imwrite(cropped_face, save_crop_path)
            # save restored face
            if args.suffix is not None:
                save_face_name = f"{basename}_{idx:02d}_{args.suffix}.png"
            else:
                save_face_name = f"{basename}_{idx:02d}.png"
            save_restore_path = os.path.join(args.output, "restored_faces", save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            if not args.no_cmp:
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(cmp_img, os.path.join(args.output, "cmp", f"{basename}_{idx:02d}.png"))

        # save restored img
        if restored_img is not None:
            if args.ext == "auto":
                extension = ext[1:]
            else:
                extension = args.ext

            if args.suffix is not None:
                save_restore_path = os.path.join(args.output, "restored_imgs", f"{basename}_{args.suffix}.{extension}")
            else:
                save_restore_path = os.path.join(args.output, "restored_imgs", f"{basename}.{extension}")
            imwrite(restored_img, save_restore_path)

    print(f"Results are in the [{args.output}] folder.")


if __name__ == "__main__":
    main()
