"""Scanner input path (linear RGB DNG) on synthetic files."""
import numpy as np
import pytest
import tifffile

from filmcolor.pipeline import convert
from filmcolor.scanner import block_means, estimate_base_scanner, is_scanner_dng, load_scanner

BASE = np.array([0.45, 0.20, 0.30])  # orange-ish base transmission, fraction of white


def scan(h=240, w=360, ir=True, white=65535):
    """Negative: base strip on the left (96 px), scene of denser patches elsewhere."""
    rng = np.random.default_rng(0)
    t = np.empty((h, w, 3))
    t[:] = BASE
    yy, xx = np.mgrid[:h, :w]
    dens = 0.15 + 1.2 * ((yy // 30 + xx // 40) % 7) / 6  # scene densities above base
    t[:, 96:] = BASE * 10 ** (-dens[:, 96:, None] * np.array([1.0, 1.1, 1.2]))
    t *= 1 + 0.002 * rng.standard_normal(t.shape)
    rgb = np.clip(np.round(t * white), 0, white).astype(np.uint16)
    irl = np.full((h, w), 64000, np.uint16) if ir else None
    return rgb, irl


def write_dng(path, rgb, irl=None, extratags=()):
    with tifffile.TiffWriter(path) as tw:
        tw.write(rgb, photometric=34892, planarconfig="contig", metadata=None, extratags=list(extratags))
        if irl is not None:  # BlueNeg marks it NewSubfileType 4; tifffile only writes that for masks
            tw.write(irl, photometric="minisblack", metadata=None)


def test_detects_linear_dng_and_rejects_other(tmp_path):
    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    assert is_scanner_dng(p)
    q = tmp_path / "rgb.dng"
    tifffile.imwrite(q, rgb, photometric="rgb")
    assert not is_scanner_dng(q)
    with pytest.raises(ValueError, match="LinearRaw"):
        load_scanner(q)
    assert not is_scanner_dng(tmp_path / "x.NEF")


def test_signal_definition_defaults_saturation_and_ir(tmp_path):
    rgb, irl = scan()
    rgb[5, 100] = 65535
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    fr = load_scanner(p)
    assert (fr.black, fr.white) == (0.0, 65535.0)
    assert fr.meta["black_source"] == "DNG default" and fr.meta["white_source"] == "DNG default"
    np.testing.assert_allclose(fr.rgb, rgb / 65535, rtol=0, atol=1e-6)
    assert fr.sat[5, 100] and fr.sat.sum() == 1
    assert fr.meta["infrared_layer"]["used"].startswith("no")
    assert fr.rgb.shape == rgb.shape  # the infrared layer is not a fourth colour channel


def test_black_and_white_level_tags(tmp_path):
    rgb, _ = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, extratags=[(50714, "H", 1, 1000, True), (50717, "H", 1, 60000, True)])
    fr = load_scanner(p)
    assert (fr.black, fr.white) == (1000.0, 60000.0)
    np.testing.assert_allclose(fr.rgb, (rgb.astype(np.float32) - 1000) / 59000, atol=1e-6)


def test_block_means_and_base_estimate(tmp_path):
    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    fr = load_scanner(p)
    sp, s = block_means(fr.rgb, fr.sat, 24)
    assert sp.shape == (10, 15, 3) and not s.any()
    np.testing.assert_allclose(sp[0, 0], fr.rgb[:24, :24].reshape(-1, 3).mean(0), rtol=1e-5)
    est = estimate_base_scanner(fr)
    np.testing.assert_allclose(est.rgb, BASE, rtol=0.01)
    assert all(c < 96 for _, c, _, _ in est.regions)  # picked from the base strip


def test_convert_end_to_end_and_replay(tmp_path):
    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    out, sc = convert(p, image_region=(20, 120, 200, 220))
    assert sc["input"]["kind"] == "scanner_linear_dng" and sc["decode"]["analysis"].startswith("same")
    np.testing.assert_allclose(sc["base"]["rgb_render"], BASE, rtol=0.01)
    assert out.shape == rgb.shape and np.all(np.isfinite(out))
    assert sc["qc"]["status"] in ("pass", "warn")
    from filmcolor.pipeline import replay_kwargs
    out2, _ = convert(p, **replay_kwargs(sc))
    assert np.array_equal(out, out2)
    out3, sc3 = convert(p, image_region=(20, 120, 200, 220), base_regions=[(0, 0, 240, 24)])
    assert sc3["base"]["method"] == "manual"
    np.testing.assert_allclose(sc3["base"]["rgb_render"], BASE, rtol=0.01)


def test_bad_image_region_fails(tmp_path):
    rgb, _ = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb)
    with pytest.raises(ValueError, match="outside"):
        convert(p, image_region=(0, 0, 1000, 10))


def test_t4l_linearization_table_is_applied(tmp_path):
    """T4l (A1283: VueScan's 8-bit Plustek DNG with a 256-entry LinearizationTable up to WhiteLevel 65535): stored
    values are looked up in the table before the black / white scaling (DNG specification); the table length is
    recorded; a decreasing table is refused."""
    rng = np.random.default_rng(4)
    a = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    lut = np.round(65535 * (np.arange(256) / 255.0) ** 2.2).astype(np.uint16)
    p = tmp_path / "s8.dng"
    write_dng(p, a, extratags=[(50712, "H", 256, lut.tolist(), True), (50717, "H", 1, 65535, True)])
    fr = load_scanner(p)
    assert fr.white == 65535.0 and fr.meta["linearization_table_len"] == 256
    np.testing.assert_allclose(fr.rgb, lut[a].astype(np.float32) / 65535, atol=1e-6)
    assert fr.sat.sum() == int((a == 255).any(-1).sum())
    q = tmp_path / "bad.dng"
    write_dng(q, a, extratags=[(50712, "H", 256, lut[::-1].tolist(), True), (50717, "H", 1, 65535, True)])
    with pytest.raises(ValueError, match="LinearizationTable"):
        load_scanner(q)
