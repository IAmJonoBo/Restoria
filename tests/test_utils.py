import pytest

# Skip this entire test module gracefully if heavy deps are not available in minimal environments
pytest.importorskip("cv2", reason="OpenCV not available in minimal test env")
pytest.importorskip("facexlib", reason="facexlib not available in minimal test env")
pytest.importorskip("gfpgan", reason="gfpgan package not available in minimal test env")


def test_gfpganer():
    # Local imports to avoid top-level import errors/lint issues when deps are missing
    import cv2  # type: ignore
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper  # type: ignore
    from gfpgan.archs.gfpganv1_arch import GFPGANv1  # type: ignore
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean  # type: ignore
    from gfpgan.utils import GFPGANer  # type: ignore
    # initialize with the clean model
    restorer = GFPGANer(
        model_path="experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth",
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    # test attribute
    assert isinstance(restorer.gfpgan, GFPGANv1Clean)
    assert isinstance(restorer.face_helper, FaceRestoreHelper)

    # initialize with the original model
    restorer = GFPGANer(
        model_path="experiments/pretrained_models/GFPGANv1.pth",
        upscale=2,
        arch="original",
        channel_multiplier=1,
        bg_upsampler=None,
    )
    # test attribute
    assert isinstance(restorer.gfpgan, GFPGANv1)
    assert isinstance(restorer.face_helper, FaceRestoreHelper)

    # ------------------ test enhance ---------------- #
    img = cv2.imread("tests/data/gt/00000000.png", cv2.IMREAD_COLOR)
    result = restorer.enhance(img, has_aligned=False, paste_back=True)
    assert result[0][0].shape == (512, 512, 3)
    assert result[1][0].shape == (512, 512, 3)
    assert result[2].shape == (1024, 1024, 3)

    # with has_aligned=True
    result = restorer.enhance(img, has_aligned=True, paste_back=False)
    assert result[0][0].shape == (512, 512, 3)
    assert result[1][0].shape == (512, 512, 3)
    assert result[2] is None
