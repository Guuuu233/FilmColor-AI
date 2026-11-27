import io

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from filmcolor.export import DESCRIPTION, icc_profile, to_uint16, write_tiff16


def lcms_to_srgb(profile_bytes, rgb8):
    """Convert 8-bit RGB through littleCMS from our profile to lcms' built-in sRGB."""
    src = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
    t = ImageCms.buildTransform(src, ImageCms.createProfile("sRGB"), "RGB", "RGB",
                                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
    img = Image.fromarray(np.array(rgb8, np.uint8).reshape(1, -1, 3), "RGB")
    return np.asarray(ImageCms.applyTransform(img, t)).reshape(-1, 3).astype(int)


@pytest.mark.parametrize("encoding", ["srgb", "linear"])
def test_icc_parses_and_describes_itself(encoding):
    prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile(encoding)))
    assert ImageCms.getProfileDescription(prof).strip() == DESCRIPTION[encoding]
    assert "uncalibrated" in DESCRIPTION[encoding]


def test_srgb_profile_matches_lcms_srgb():
    colours = [(0, 0, 0), (255, 255, 255), (128, 128, 128), (200, 30, 40), (20, 180, 60), (30, 60, 220)]
    out = lcms_to_srgb(icc_profile("srgb"), colours)
    assert np.max(np.abs(out - np.array(colours))) <= 2


def test_linear_profile_has_identity_transfer():
    out = lcms_to_srgb(icc_profile("linear"), [(128, 128, 128), (255, 255, 255), (0, 0, 0)])
    np.testing.assert_allclose(out[0], 188, atol=2)  # linear 0.502 -> sRGB 0.737
    np.testing.assert_allclose(out[1:], [[255] * 3, [0] * 3], atol=1)


def test_icc_is_deterministic_and_size_consistent():
    a, b = icc_profile("srgb"), icc_profile("srgb")
    assert a == b
    assert int.from_bytes(a[:4], "big") == len(a)
    assert a[36:40] == b"acsp"
    with pytest.raises(ValueError):
        icc_profile("prophoto")


def test_tiff_roundtrip_bit_depth_icc_and_values(tmp_path):
    img = np.random.default_rng(0).uniform(-0.1, 1.1, (40, 60, 3)).astype(np.float32)
    p = tmp_path / "x.tif"
    write_tiff16(p, img, "srgb")
    with tifffile.TiffFile(p) as tf:
        page = tf.pages[0]
        data = page.asarray()
        assert data.dtype == np.uint16 and data.shape == (40, 60, 3)
        assert page.photometric == tifffile.PHOTOMETRIC.RGB
        assert bytes(page.tags["InterColorProfile"].value) == icc_profile("srgb")
    assert np.array_equal(data, to_uint16(img))
    assert data.min() == 0 and data.max() == 65535  # clipped, not wrapped


def test_tiff_bytes_are_reproducible(tmp_path):
    img = np.random.default_rng(1).uniform(0, 1, (32, 32, 3)).astype(np.float32)
    write_tiff16(tmp_path / "a.tif", img)
    write_tiff16(tmp_path / "b.tif", img)
    assert (tmp_path / "a.tif").read_bytes() == (tmp_path / "b.tif").read_bytes()


def test_tiff_rejects_bad_shape(tmp_path):
    with pytest.raises(ValueError):
        write_tiff16(tmp_path / "x.tif", np.zeros((10, 10), np.float32))


def test_p10_to_uint16_chunked_bit_exact():
    """P10: chunked to_uint16 (>512 rows) is bit-exact with unchunked reference."""
    rng = np.random.default_rng(42)
    img = rng.uniform(-0.1, 1.1, (600, 17, 3)).astype(np.float32)
    expected = np.round(np.clip(img, 0, 1) * 65535).astype(np.uint16)
    got = to_uint16(img)
    np.testing.assert_array_equal(got, expected)
