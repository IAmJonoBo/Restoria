import os

import torch
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize


class CodeFormerRestorer:
    """Minimal wrapper to run CodeFormer as a backend.

    Requires the CodeFormer repo installed (see https://github.com/sczhou/CodeFormer).
    """

    def __init__(self, model_path: str, device: torch.device, upscale: int = 2, bg_upsampler=None):
        self.device = device
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        try:
            from codeformer.archs.codeformer_arch import CodeFormer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "CodeFormer backend requires the CodeFormer repo installed: "
                "pip install --no-cache-dir git+https://github.com/sczhou/CodeFormer@master"
            ) from e

        # Initialize model
        self.codeformer = CodeFormer().to(self.device)
        # Load the checkpoint
        loadnet = torch.load(model_path, map_location=self.device, weights_only=True)
        key = "params_ema" if "params_ema" in loadnet else "params"
        self.codeformer.load_state_dict(loadnet[key], strict=True)
        self.codeformer.eval()

        # Face helper for alignment and paste-back
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath=os.path.join("gfpgan", "weights"),
        )

    @torch.no_grad()
    def enhance(self, img_bgr, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5):
        self.face_helper.clean_all()

        if has_aligned:
            import cv2

            img_bgr = cv2.resize(img_bgr, (512, 512))
            self.face_helper.cropped_faces = [img_bgr]
        else:
            self.face_helper.read_image(img_bgr)
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            self.face_helper.align_warp_face()

        restored_faces = []
        for cropped_face in self.face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                # CodeFormer may accept a fidelity weight; pass via keyword if supported
                if "w" in self.codeformer.forward.__code__.co_varnames:
                    out = self.codeformer(cropped_face_t, w=weight)
                else:
                    out = self.codeformer(cropped_face_t)
                face = tensor2img(out.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except Exception:  # pragma: no cover
                # Fallback to input if inference fails
                face = cropped_face
            face = face.astype("uint8")
            restored_faces.append(face)
            self.face_helper.add_restored_face(face)

        if not has_aligned and paste_back:
            if self.bg_upsampler is not None:
                bg_img = self.bg_upsampler.enhance(img_bgr, outscale=self.upscale)[0]
            else:
                bg_img = None
            self.face_helper.get_inverse_affine(None)
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, restored_faces, None
