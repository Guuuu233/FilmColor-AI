import io
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from filmcolor.colorimetry import (MatrixTRCProfile, delta_e2000, lab_to_xyz, xyz_to_lab,
                                   xyz_to_srgb_preview)
from filmcolor.compare import (align, aligned_pair, apply_ab_affine, block_mean, decompose, fit_ab_affine,
                               load_lab, match_tone, metrics, ncc_map, resize_bilinear)
from filmcolor.export import icc_profile, write_tiff16

NLP_REF = Path(__file__).resolve().parents[1] / "data/reference/nlp_default/_DSF8948-positive.tif"


def test_delta_e2000_sharma_reference_pairs():
    # Sharma, Wu, Dalal (2005), test data pairs 1, 2, 3 and 7
    lab1 = np.array([[50, 2.6772, -79.7751], [50, 3.1571, -77.2803], [50, 2.8361, -74.0200], [50, 0, 0]])
    lab2 = np.array([[50, 0, -82.7485], [50, 0, -82.7485], [50, 0, -82.7485], [50, -1, 2]])
    np.testing.assert_allclose(delta_e2000(lab1, lab2), [2.0425, 2.8615, 3.4412, 2.3669], atol=1e-4)
    assert delta_e2000(lab1[:1], lab1[:1])[0] == 0


def test_lab_roundtrip():
    xyz = np.random.default_rng(0).uniform(0.001, 1, (100, 3))
    np.testing.assert_allclose(lab_to_xyz(xyz_to_lab(xyz)), xyz, rtol=1e-9)


def lcms_srgb(icc, rgb8):
    t = ImageCms.buildTransform(ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile("sRGB"),
                                "RGB", "RGB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    img = Image.fromarray(np.asarray(rgb8, np.uint8).reshape(1, -1, 3), "RGB")
    return np.asarray(ImageCms.applyTransform(img, t)).reshape(-1, 3).astype(int)


COLOURS = [(0, 0, 0), (255, 255, 255), (128, 128, 128), (200, 30, 40), (20, 180, 60), (30, 60, 220), (240, 200, 120)]


def ours_srgb(icc, rgb8):
    xyz = MatrixTRCProfile(icc).to_xyz(np.asarray(rgb8, float) / 255)
    return np.round(xyz_to_srgb_preview(xyz) * 255).astype(int)


@pytest.mark.parametrize("encoding", ["srgb", "linear"])
def test_profile_parser_matches_lcms_on_our_profiles(encoding):
    icc = icc_profile(encoding)
    assert np.max(np.abs(ours_srgb(icc, COLOURS) - lcms_srgb(icc, COLOURS))) <= 2


@pytest.mark.skipif(not NLP_REF.exists(), reason="NLP reference TIFFs not present")
def test_profile_parser_matches_lcms_on_lightroom_prophoto():
    icc = bytes(tifffile.TiffFile(NLP_REF).pages[0].tags["InterColorProfile"].value)
    in_gamut = [(128, 128, 128), (255, 255, 255), (0, 0, 0), (150, 110, 90), (90, 110, 130), (120, 140, 100)]
    assert np.max(np.abs(ours_srgb(icc, in_gamut) - lcms_srgb(icc, in_gamut))) <= 2


def test_load_lab_reads_embedded_profile(tmp_path):
    img = np.zeros((4, 4, 3), np.float32)
    img[:2] = 1.0
    write_tiff16(tmp_path / "x.tif", img, "srgb")
    lab = load_lab(tmp_path / "x.tif")
    np.testing.assert_allclose(lab[0, 0], [100, 0, 0], atol=0.05)
    np.testing.assert_allclose(lab[3, 3], [0, 0, 0], atol=0.05)
    tifffile.imwrite(tmp_path / "noicc.tif", np.zeros((4, 4, 3), np.uint16), photometric="rgb")
    with pytest.raises(ValueError, match="ICC"):
        load_lab(tmp_path / "noicc.tif")


def test_load_lab_reads_jpeg_with_embedded_profile(tmp_path):
    img = np.zeros((48, 16, 3), np.uint8)
    img[:16], img[16:32] = 255, 128
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(img).save(tmp_path / "x.jpg", quality=100, subsampling=0, icc_profile=icc)
    lab = load_lab(tmp_path / "x.jpg")
    np.testing.assert_allclose(lab[8, 8], [100, 0, 0], atol=0.3)
    np.testing.assert_allclose(lab[24, 8], [53.59, 0, 0], atol=0.3)  # sRGB 128 grey
    np.testing.assert_allclose(lab[40, 8], [0, 0, 0], atol=0.3)
    Image.fromarray(img).save(tmp_path / "noicc.jpg")
    with pytest.raises(ValueError, match="ICC"):
        load_lab(tmp_path / "noicc.jpg")


def smooth_scene(h=480, w=720, seed=0):
    rng = np.random.default_rng(seed)
    small = rng.uniform(0, 100, (h // 24, w // 24))
    return resize_bilinear(small, (h, w)) + rng.normal(0, 0.5, (h, w))


def test_ncc_finds_exact_template_position():
    img = smooth_scene()
    m = ncc_map(img, img[100:300, 200:450])
    assert np.unravel_index(np.argmax(m), m.shape) == (100, 200)
    assert m.max() == pytest.approx(1, abs=1e-6)


def test_align_recovers_rotation_scale_and_offset():
    img = smooth_scene()
    crop = img[80:400, 160:560]                                   # 320 x 400 at full res
    ref = np.rot90(resize_bilinear(crop, (323, 404)), -1)         # rescaled ~1%, rotated clockwise
    al = align(img, ref, factor=4)
    assert al.rot90 == 1
    assert abs(al.scale - 0.99) <= 0.006
    assert abs(al.offset[0] - 20) <= 1 and abs(al.offset[1] - 40) <= 1
    assert al.ncc > 0.95


def test_metrics_on_identical_and_shifted_images():
    lab = np.stack(np.meshgrid(np.linspace(20, 80, 50), np.linspace(-20, 20, 40))[::-1], -1)
    lab = np.concatenate([lab[..., :1], lab[..., 1:2], np.full(lab.shape[:2] + (1,), 15.0)], -1)
    same = metrics(lab, lab)
    assert same["de2000_median"] == 0 and same["mean_dL"] == 0
    brighter = lab.copy()
    brighter[..., 0] += 5
    assert metrics(brighter, lab)["mean_dL"] == pytest.approx(5)


def test_aligned_pair_shapes_match():
    img = np.repeat(smooth_scene()[..., None], 3, -1)
    ref = img[80:400, 160:560]
    al = align(img[..., 0], ref[..., 0], factor=4)
    a, r = aligned_pair(img, ref, al)
    assert a.shape == r.shape
    assert metrics(a, r)["de2000_median"] < 0.5


def test_load_lab_averages_in_linear_xyz(tmp_path):
    img = np.zeros((4, 4, 3), np.float32)
    img[:, ::2] = 1.0  # alternating white / black columns
    write_tiff16(tmp_path / "x.tif", img, "linear")
    lab = load_lab(tmp_path / "x.tif", factor=2)
    assert lab.shape == (2, 2, 3)
    np.testing.assert_allclose(lab[..., 0], xyz_to_lab(np.array([[0.5 * 0.9642, 0.5, 0.5 * 0.8249]]))[0, 0], atol=0.05)


def test_match_tone_reproduces_reference_distribution():
    rng = np.random.default_rng(0)
    L, ref = rng.uniform(0, 50, 1000), rng.uniform(30, 90, 1000)
    out = match_tone(L, ref)
    assert np.all(np.diff(out[np.argsort(L)]) >= 0)  # monotone
    np.testing.assert_allclose(np.sort(out), np.quantile(ref, (np.arange(1000) + 0.5) / 1000))


def test_ab_affine_recovers_known_map():
    rng = np.random.default_rng(1)
    ab = rng.uniform(-40, 40, (500, 2))
    coef = np.array([[1.3, 0.2], [-0.1, 0.9], [2.0, -3.0]])
    np.testing.assert_allclose(fit_ab_affine(ab, apply_ab_affine(ab, coef)), coef, atol=1e-9)


def test_decompose_explains_global_differences():
    rng = np.random.default_rng(2)
    ref = np.stack([rng.uniform(10, 90, (40, 50)), rng.uniform(-30, 30, (40, 50)), rng.uniform(-30, 30, (40, 50))], -1)
    ours = ref.copy()
    ours[..., 0] = 100 * (ref[..., 0] / 100) ** 1.5                     # different tone curve
    ours[..., 1:] = 0.5 * ref[..., 1:] + np.array([3.0, -4.0])         # desaturated, cast
    d = decompose(ours, ref)
    assert d["none"] > 5 and d["tone"] < d["none"] and d["tone_ab"] < 0.5
