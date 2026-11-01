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


def test_t4m_interleaved_4ch_rgbi_compatibility(tmp_path):
    """T4m: 4-channel RGBI linear DNG is accepted by _main_page/load_scanner; RGB signal, sat mask,
    base estimation, and end-to-end convert match the 3-channel version identically."""
    rgb, _ = scan()
    p3 = tmp_path / "s3.dng"
    write_dng(p3, rgb)

    ir_ch = np.linspace(30000, 60000, rgb.shape[0] * rgb.shape[1], dtype=np.uint16).reshape(rgb.shape[:2])
    rgbi = np.dstack([rgb, ir_ch])
    p4 = tmp_path / "s4.dng"
    write_dng(p4, rgbi)

    assert is_scanner_dng(p4)
    fr3 = load_scanner(p3)
    fr4 = load_scanner(p4)

    assert fr4.rgb.shape == (rgb.shape[0], rgb.shape[1], 3)
    np.testing.assert_array_equal(fr4.rgb, fr3.rgb)
    np.testing.assert_array_equal(fr4.sat, fr3.sat)
    assert (fr4.black, fr4.white) == (fr3.black, fr3.white)

    # Metadata records interleaved_4ch source
    ir_meta = fr4.meta["infrared_layer"]
    assert ir_meta is not None
    assert ir_meta["source"] == "interleaved_4ch"
    assert ir_meta["used"] == "no (recorded only)"
    assert np.isclose(ir_meta["percentiles"]["50"], np.percentile(ir_ch[::8, ::8], 50))

    # Base estimation matches identically
    est3 = estimate_base_scanner(fr3)
    est4 = estimate_base_scanner(fr4)
    np.testing.assert_allclose(est4.rgb, est3.rgb)
    assert est4.regions == est3.regions

    # Convert matches identically
    out3, sc3 = convert(p3, image_region=(20, 120, 200, 220))
    out4, sc4 = convert(p4, image_region=(20, 120, 200, 220))
    np.testing.assert_array_equal(out4, out3)
    np.testing.assert_allclose(sc4["base"]["rgb_render"], sc3["base"]["rgb_render"])


def test_t4m_interleaved_ir_does_not_affect_rgb_sat_or_base(tmp_path):
    """T4m: extreme values in the 4th (IR) channel do not affect RGB saturation mask or base estimation."""
    rgb, _ = scan()
    rgb[10, 20, 0] = 65535  # exactly one saturated pixel in RGB
    # IR channel fully saturated (65535) everywhere
    rgbi = np.dstack([rgb, np.full(rgb.shape[:2], 65535, dtype=np.uint16)])
    p = tmp_path / "sat_ir.dng"
    write_dng(p, rgbi)

    fr = load_scanner(p)
    # fr.sat must only reflect RGB channels (exactly 1 saturated pixel, not the whole frame)
    assert fr.sat.sum() == 1
    assert fr.sat[10, 20]
    assert fr.meta["infrared_layer"]["percentiles"]["50"] == 65535.0

    est = estimate_base_scanner(fr)
    np.testing.assert_allclose(est.rgb, BASE, rtol=0.01)


def test_t4m_black_and_white_level_tags_4ch(tmp_path):
    """T4m: 4-sample BlackLevel and WhiteLevel tags are accepted when uniform; non-uniform rejected."""
    rgb, _ = scan()
    rgbi = np.dstack([rgb, np.full(rgb.shape[:2], 50000, dtype=np.uint16)])
    p = tmp_path / "levels4.dng"
    write_dng(p, rgbi, extratags=[
        (50714, "H", 4, [1000, 1000, 1000, 1000], True),
        (50717, "H", 4, [60000, 60000, 60000, 60000], True),
    ])
    fr = load_scanner(p)
    assert (fr.black, fr.white) == (1000.0, 60000.0)
    np.testing.assert_allclose(fr.rgb, (rgb.astype(np.float32) - 1000) / 59000, atol=1e-6)

    # Non-uniform BlackLevel must be rejected
    p_bad = tmp_path / "bad_levels4.dng"
    write_dng(p_bad, rgbi, extratags=[
        (50714, "H", 4, [1000, 1000, 2000, 1000], True),
        (50717, "H", 4, [60000, 60000, 60000, 60000], True),
    ])
    with pytest.raises(ValueError, match="per-sample"):
        load_scanner(p_bad)


def test_t4m_separate_ir_page_priority_over_interleaved(tmp_path):
    """T4m: when both a separate 1-sample IR page and an interleaved 4th channel exist, the separate page takes priority."""
    rgb, _ = scan()
    rgbi = np.dstack([rgb, np.full(rgb.shape[:2], 20000, dtype=np.uint16)])  # channel 4 has 20000
    irl = np.full(rgb.shape[:2], 55000, dtype=np.uint16)  # separate IR page has 55000
    p = tmp_path / "both_ir.dng"
    write_dng(p, rgbi, irl=irl)

    fr = load_scanner(p)
    ir_meta = fr.meta["infrared_layer"]
    assert "source" not in ir_meta  # separate page uses new_subfile_type, not source=interleaved_4ch
    assert "new_subfile_type" in ir_meta
    assert np.isclose(ir_meta["percentiles"]["50"], 55000.0)


def test_t4m_interleaved_4ch_linearization_table(tmp_path):
    """T4m: LinearizationTable path works on 4-channel RGBI."""
    rng = np.random.default_rng(42)
    a4 = rng.integers(0, 256, (120, 160, 4), dtype=np.uint8)
    lut = np.round(65535 * (np.arange(256) / 255.0) ** 2.2).astype(np.uint16)
    p = tmp_path / "s8_4ch.dng"
    write_dng(p, a4, extratags=[(50712, "H", 256, lut.tolist(), True), (50717, "H", 1, 65535, True)])

    fr = load_scanner(p)
    assert fr.white == 65535.0 and fr.meta["linearization_table_len"] == 256
    np.testing.assert_allclose(fr.rgb, lut[a4[:, :, :3]].astype(np.float32) / 65535, atol=1e-6)
    assert fr.meta["infrared_layer"]["source"] == "interleaved_4ch"
    # IR stats reflect linearized samples
    ir_linearized = lut[a4[::8, ::8, 3]]
    assert np.isclose(fr.meta["infrared_layer"]["percentiles"]["50"], np.percentile(ir_linearized, 50))


def test_t4p_monochrome_single_channel_identification_and_equal_rgb(tmp_path):
    """T4p: 1-channel LinearRaw (H, W) or (H, W, 1) is identified as scanner DNG and loaded as equal 3-channel RGB.
    monochrome_dng=True is set in meta (and omitted for RGB/RGBI); saturation mask and levels match."""
    rng = np.random.default_rng(101)
    m = rng.integers(1000, 60000, (120, 160), dtype=np.uint16)
    m[10, 20] = 65535  # saturated pixel
    p_mono = tmp_path / "mono_2d.dng"
    with tifffile.TiffWriter(p_mono) as tw:
        tw.write(m, photometric=34892)

    assert is_scanner_dng(p_mono)
    fr = load_scanner(p_mono)
    assert fr.meta.get("monochrome_dng") is True
    assert fr.rgb.shape == (120, 160, 3)
    # Equal RGB across all 3 channels
    np.testing.assert_array_equal(fr.rgb[:, :, 0], fr.rgb[:, :, 1])
    np.testing.assert_array_equal(fr.rgb[:, :, 1], fr.rgb[:, :, 2])
    np.testing.assert_allclose(fr.rgb[:, :, 0], m.astype(np.float32) / 65535.0, atol=1e-6)
    # Saturation mask
    assert fr.sat.shape == (120, 160)
    assert fr.sat[10, 20] and fr.sat.sum() == 1
    # Levels
    assert (fr.black, fr.white) == (0.0, 65535.0)
    # IR list should exclude main itself
    assert fr.meta["infrared_layer"] is None

    # (H, W, 1) 3D single channel
    p_mono3d = tmp_path / "mono_3d.dng"
    with tifffile.TiffWriter(p_mono3d) as tw:
        tw.write(m[:, :, None], photometric=34892, planarconfig="contig")
    fr3d = load_scanner(p_mono3d)
    assert fr3d.meta.get("monochrome_dng") is True
    np.testing.assert_array_equal(fr3d.rgb, fr.rgb)
    np.testing.assert_array_equal(fr3d.sat, fr.sat)


def test_t4p_monochrome_levels_and_lut(tmp_path):
    """T4p: single-channel BlackLevel/WhiteLevel and LinearizationTable apply correctly."""
    rng = np.random.default_rng(202)
    m = rng.integers(0, 256, (80, 100), dtype=np.uint8)
    lut = np.round(65535 * (np.arange(256) / 255.0) ** 2.0).astype(np.uint16)
    p = tmp_path / "mono_lut.dng"
    with tifffile.TiffWriter(p) as tw:
        tw.write(m, photometric=34892, extratags=[
            (50712, "H", 256, lut.tolist(), True),
            (50714, "H", 1, 500, True),
            (50717, "H", 1, 62000, True),
        ])

    fr = load_scanner(p)
    assert fr.meta.get("monochrome_dng") is True
    assert fr.meta["linearization_table_len"] == 256
    assert (fr.black, fr.white) == (500.0, 62000.0)
    expected_linear = lut[m].astype(np.float32)
    expected_rgb_ch = (expected_linear - 500.0) / (62000.0 - 500.0)
    np.testing.assert_allclose(fr.rgb[:, :, 0], expected_rgb_ch, atol=1e-6)
    np.testing.assert_array_equal(fr.rgb[:, :, 0], fr.rgb[:, :, 1])
    np.testing.assert_array_equal(fr.rgb[:, :, 1], fr.rgb[:, :, 2])
    assert np.array_equal(fr.sat, expected_linear >= 62000.0)


def test_t4p_monochrome_with_separate_ir_layer(tmp_path):
    """T4p: monochrome main page with a separate 1-sample IR page extracts IR stats without self-exclusion error."""
    m = np.full((100, 100), 20000, dtype=np.uint16)
    ir_data = np.full((100, 100), 45000, dtype=np.uint16)
    p = tmp_path / "mono_with_ir.dng"
    with tifffile.TiffWriter(p) as tw:
        tw.write(m, photometric=34892)
        tw.write(ir_data, photometric="minisblack")

    fr = load_scanner(p)
    assert fr.meta.get("monochrome_dng") is True
    ir_meta = fr.meta["infrared_layer"]
    assert ir_meta is not None
    assert np.isclose(ir_meta["percentiles"]["50"], 45000.0)
    assert ir_meta["used"].startswith("no")


def test_t4p_rgb_priority_over_monochrome_main_page(tmp_path):
    """T4p: when both RGB (samplesperpixel=3) and monochrome (samplesperpixel=1) LinearRaw pages exist,
    RGB is selected as main page regardless of page order."""
    rgb, _ = scan(100, 100)
    mono = np.full((100, 100), 12345, dtype=np.uint16)

    # Order 1: mono first, then rgb
    p1 = tmp_path / "mono_then_rgb.dng"
    with tifffile.TiffWriter(p1) as tw:
        tw.write(mono, photometric=34892)
        tw.write(rgb, photometric=34892, planarconfig="contig")
    fr1 = load_scanner(p1)
    assert "monochrome_dng" not in fr1.meta
    np.testing.assert_allclose(fr1.rgb, rgb / 65535.0, atol=1e-6)

    # Order 2: rgb first, then mono (mono acts as IR candidate if same shape)
    p2 = tmp_path / "rgb_then_mono.dng"
    with tifffile.TiffWriter(p2) as tw:
        tw.write(rgb, photometric=34892, planarconfig="contig")
        tw.write(mono, photometric=34892)
    fr2 = load_scanner(p2)
    assert "monochrome_dng" not in fr2.meta
    np.testing.assert_allclose(fr2.rgb, rgb / 65535.0, atol=1e-6)
