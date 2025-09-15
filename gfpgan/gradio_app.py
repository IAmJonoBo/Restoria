import argparse
import os
from typing import Optional

import gradio as gr


def build_restorer(
    version: str,
    upscale: int,
    device: str,
    bg_upsampler: str,
    bg_tile: int,
    bg_precision: str,
    detector: str,
    use_parse: bool,
    channel_multiplier: int = 2,
):
    # Lazy imports to avoid heavy startup
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    from gfpgan import GFPGANer

    # Resolve model metadata
    if version == "1":
        arch = "original"
        channel_multiplier = 1
        model_name = "GFPGANv1"
        url = "https://huggingface.co/TencentARC/GFPGANv1/resolve/main/GFPGANv1.pth"
    elif version == "1.2":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANCleanv1-NoCE-C2"
        url = "https://huggingface.co/TencentARC/GFPGANv1/resolve/main/GFPGANCleanv1-NoCE-C2.pth"
    elif version == "1.3":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANv1.3"
        url = "https://huggingface.co/TencentARC/GFPGANv1/resolve/main/GFPGANv1.3.pth"
    elif version == "1.4":
        arch = "clean"
        channel_multiplier = 2
        model_name = "GFPGANv1.4"
        url = "https://huggingface.co/TencentARC/GFPGANv1/resolve/main/GFPGANv1.4.pth"
    elif version == "RestoreFormer":
        arch = "RestoreFormer"
        channel_multiplier = 2
        model_name = "RestoreFormer"
        url = "https://huggingface.co/TencentARC/GFPGANv1/resolve/main/RestoreFormer.pth"
    else:
        raise ValueError(f"Unknown version: {version}")

    # Resolve weight path (prefer local)
    model_path = os.path.join("experiments/pretrained_models", model_name + ".pth")
    if not os.path.isfile(model_path):
        alt = os.path.join("gfpgan/weights", model_name + ".pth")
        model_path = model_path if os.path.isfile(model_path) else (alt if os.path.isfile(alt) else url)

    # Device
    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else device.replace("auto", "cpu")
    dev_t = torch.device(dev)

    # Background upsampler
    if bg_upsampler == "realesrgan" and dev == "cuda":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        half = True if bg_precision == "auto" else (bg_precision == "fp16")
        bg = RealESRGANer(
            scale=2,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            model=model,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=half,
        )
    else:
        bg = None

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg,
        device=dev_t,
        det_model=detector,
        use_parse=use_parse,
    )
    return restorer, dev


def app_main():
    with gr.Blocks(title="GFPGAN Local UI") as demo:
        gr.Markdown("# GFPGAN — Local Demo")
        with gr.Row():
            with gr.Column(scale=1):
                version = gr.Dropdown(["1", "1.2", "1.3", "1.4", "RestoreFormer"], value="1.4", label="Version")
                device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto", label="Device")
                upscale = gr.Slider(1, 4, value=2, step=1, label="Upscale")
                weight = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Weight")
                detector = gr.Dropdown(
                    ["retinaface_resnet50", "retinaface_mobile0.25", "scrfd"],
                    value="retinaface_resnet50",
                    label="Detector",
                )
                parse = gr.Checkbox(value=True, label="Use face parsing")
                bg_upsampler = gr.Dropdown(["realesrgan", "none"], value="realesrgan", label="BG upsampler")
                bg_precision = gr.Dropdown(["auto", "fp16", "fp32"], value="auto", label="BG precision")
                bg_tile = gr.Slider(0, 800, value=400, step=50, label="BG tile")
                input_imgs = gr.Image(
                    type="numpy",
                    label="Input Image(s)",
                    sources=["upload", "clipboard"],
                    tool=None,
                    image_mode="RGB",
                    multiple=True,
                )
                run_btn = gr.Button("Run")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Restored Images", show_label=True)
                logs = gr.Markdown()

        def infer(images, version, device, upscale, weight, detector, parse, bg_upsampler, bg_precision, bg_tile):
            import cv2

            if not images:
                return [], "No images provided."
            restorer, dev = build_restorer(
                version,
                int(upscale),
                device,
                bg_upsampler,
                int(bg_tile),
                bg_precision,
                detector,
                bool(parse),
            )
            outs = []
            for idx, img in enumerate(images):
                # images come as RGB; convert to BGR for OpenCV path
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                _, faces, restored = restorer.enhance(
                    bgr,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=float(weight),
                )
                if restored is None:
                    # fallback to face tiles if no paste_back
                    if faces:
                        outs.append(cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB))
                else:
                    outs.append(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
            return outs, f"Device: {dev}; Processed {len(outs)} images"

        run_btn.click(
            infer,
            inputs=[input_imgs, version, device, upscale, weight, detector, parse, bg_upsampler, bg_precision, bg_tile],
            outputs=[gallery, logs],
        )

    return demo


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Run local Gradio demo for GFPGAN")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args(argv)

    demo = app_main()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
