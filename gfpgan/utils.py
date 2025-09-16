import os
import importlib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _warn(msg: str) -> None:
    """Best-effort warning: prefer logging; fallback to print.

    Keeps import-time light and avoids adding hard logging deps.
    """
    try:
        import logging  # standard lib

        logging.getLogger("gfpgan").warning(msg)
        return
    except Exception:
        pass
    try:
        print(f"[WARN] {msg}")
    except Exception:
        pass


class GFPGANer:
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(
        self,
        model_path,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=None,
        det_model: str = "retinaface_resnet50",
        use_parse: bool = True,
    ):
        # Defer heavy imports here to keep module import light
        import torch  # local import

        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # Lazy-load architecture based on selection
        if arch == "clean":
            # Intentional dynamic import to keep top-level import cost low
            arch_cls = getattr(
                importlib.import_module("gfpgan.archs.gfpganv1_clean_arch"),
                "GFPGANv1Clean",
            )
            self.gfpgan = arch_cls(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "bilinear":
            # Intentional dynamic import to keep top-level import cost low
            arch_cls = getattr(
                importlib.import_module("gfpgan.archs.gfpgan_bilinear_arch"),
                "GFPGANBilinear",
            )
            self.gfpgan = arch_cls(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "original":
            # Intentional dynamic import to keep top-level import cost low
            arch_cls = getattr(
                importlib.import_module("gfpgan.archs.gfpganv1_arch"),
                "GFPGANv1",
            )
            self.gfpgan = arch_cls(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "RestoreFormer":
            # Intentional dynamic import to keep top-level import cost low
            arch_cls = getattr(
                importlib.import_module("gfpgan.archs.restoreformer_arch"),
                "RestoreFormer",
            )
            self.gfpgan = arch_cls()

        # initialize face helper
        # Intentional dynamic import to keep top-level import cost low
        face_restore_helper_cls = getattr(
            importlib.import_module("facexlib.utils.face_restoration_helper"),
            "FaceRestoreHelper",
        )
        self.face_helper = face_restore_helper_cls(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=det_model,
            save_ext="png",
            use_parse=use_parse,
            device=self.device,
            model_rootpath="gfpgan/weights",
        )

        # Resolve URL weights if needed
        if model_path.startswith("https://"):
            # Intentional dynamic import to keep top-level import cost low
            download_fn = getattr(
                importlib.import_module("basicsr.utils.download_util"),
                "load_file_from_url",
            )
            model_path = download_fn(
                url=model_path, model_dir=os.path.join(ROOT_DIR, "gfpgan/weights"), progress=True, file_name=None
            )

        # Map weights to the selected device to avoid CUDA-only loads on CPU
        # Prefer safer weights-only deserialization when available (PyTorch >= 2.0)
        try:
            loadnet = torch.load(model_path, map_location=self.device, weights_only=True)  # type: ignore[call-arg]
        except TypeError:
            # Older torch versions do not support weights_only
            # Security note: full deserialization is less safe; prefer upgrading to torch>=2.0
            _warn(
                "torch.load(weights_only=True) unsupported by this torch version; "
                "falling back to full deserialization. Consider upgrading torch for safer loads."
            )
            loadnet = torch.load(model_path, map_location=self.device)
        keyname = "params_ema" if "params_ema" in loadnet else "params"
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    def enhance(
        self,
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.5,
        eye_dist_threshold=5,
    ):
        # Localize heavy imports
        import torch  # local import
        import cv2  # local import
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            # Primary detection
            self.face_helper.read_image(img)
            self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, eye_dist_threshold=eye_dist_threshold
            )
            # align and warp each face
            self.face_helper.align_warp_face()
            # Fallback detectors if none found
            if len(self.face_helper.cropped_faces) == 0:
                tried = {getattr(self.face_helper, "det_model", "retinaface_resnet50")}
                for alt in ("scrfd", "retinaface_resnet50", "retinaface_mobile0.25"):
                    if alt in tried:
                        continue
                    try:
                        self.face_helper.clean_all()
                        self.face_helper.read_image(img)
                        # switch detector and retry
                        self.face_helper.det_model = alt
                        self.face_helper.get_face_landmarks_5(
                            only_center_face=only_center_face, eye_dist_threshold=eye_dist_threshold
                        )
                        self.face_helper.align_warp_face()
                        if len(self.face_helper.cropped_faces) > 0:
                            break
                    except Exception:
                        continue

        # face restoration
        with torch.no_grad():
            for cropped_face in self.face_helper.cropped_faces:
                # prepare data
                cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

                try:
                    output = self.gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0]
                    # convert to image
                    restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                except RuntimeError as error:
                    print(f"\tFailed inference for GFPGAN: {error}.")
                    restored_face = cropped_face

                restored_face = restored_face.astype("uint8")
                self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
