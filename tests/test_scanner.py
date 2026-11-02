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


# ---------------------------------------------------------------------------
# T4s: 16-bit scanner TIFF support, LZW/PackBits/Predictor2 fallback & guards
# ---------------------------------------------------------------------------

def encode_packbits(data: bytes) -> bytes:
    """Encode bytes using TIFF PackBits run-length compression."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        # Check for repeated bytes
        run_len = 1
        while i + run_len < n and data[i + run_len] == data[i] and run_len < 128:
            run_len += 1
        if run_len >= 3 or (run_len == 2 and i + 2 == n):
            out.append(257 - run_len)
            out.append(data[i])
            i += run_len
        else:
            # Literal run
            lit_start = i
            while i < n and (i - lit_start < 128):
                if i + 2 < n and data[i] == data[i + 1] == data[i + 2]:
                    break
                i += 1
            lit_count = i - lit_start
            out.append(lit_count - 1)
            out.extend(data[lit_start:i])
    return bytes(out)


def encode_tiff_lzw(data: bytes, *, include_eoi: bool = True) -> bytes:
    """Encode bytes using TIFF LZW with MSB-first bit-packing and early change."""
    table = {bytes([b]): b for b in range(256)}
    next_code = 258
    code_size = 9
    out_bits = []

    def put_code(c, bits):
        for b in range(bits - 1, -1, -1):
            out_bits.append((c >> b) & 1)

    put_code(256, code_size)  # ClearCode
    s = b""
    for byte in data:
        c = bytes([byte])
        sc = s + c
        if sc in table:
            s = sc
        else:
            put_code(table[s], code_size)
            if next_code < 4096:
                table[sc] = next_code
                next_code += 1
                if next_code == (1 << code_size) and code_size < 12:
                    code_size += 1
            s = c
    if s:
        put_code(table[s], code_size)
    if include_eoi:
        put_code(257, code_size)  # EOI

    pad = (8 - len(out_bits) % 8) % 8
    out_bits.extend([0] * pad)
    out_bytes = bytearray()
    for j in range(0, len(out_bits), 8):
        val = 0
        for bit in out_bits[j:j + 8]:
            val = (val << 1) | bit
        out_bytes.append(val)
    return bytes(out_bytes)


def build_synthetic_tiff(strips_data: list[bytes], width: int, height: int, spp: int = 3,
                         photometric: int = 2, compression: int = 1, predictor: int = 1,
                         planarconfig: int = 1, rows_per_strip: int | None = None,
                         byteorder: str = "<") -> bytes:
    """Build a valid standalone TIFF in memory with custom strip data and compression."""
    import struct
    if rows_per_strip is None:
        rows_per_strip = height
    bo = byteorder
    header = b"II\x2a\x00" if bo == "<" else b"MM\x00\x2a"

    offset = 8
    strip_offsets = []
    strip_bytecounts = []
    all_data = b""
    for s in strips_data:
        strip_offsets.append(offset + len(all_data))
        strip_bytecounts.append(len(s))
        all_data += s

    ifd_offset = offset + len(all_data)
    if ifd_offset % 2 == 1:
        all_data += b"\x00"
        ifd_offset += 1

    entries = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, spp, [16] * spp),
        (259, 3, 1, compression),
        (262, 3, 1, photometric),
        (273, 4, len(strip_offsets), strip_offsets),
        (277, 3, 1, spp),
        (278, 4, 1, rows_per_strip),
        (279, 4, len(strip_bytecounts), strip_bytecounts),
    ]
    if planarconfig != 1:
        entries.append((284, 3, 1, planarconfig))
    if predictor != 1:
        entries.append((317, 3, 1, predictor))

    entries.sort(key=lambda x: x[0])
    ifd_len = 2 + len(entries) * 12 + 4
    extra_data_offset = ifd_offset + ifd_len

    ifd_bytes = bytearray(struct.pack(f"{bo}H", len(entries)))
    extra_data = bytearray()

    for tag, type_id, count, val in entries:
        if type_id == 3:
            raw = struct.pack(f"{bo}{count}H", *(val if isinstance(val, (list, tuple)) else [val]))
        elif type_id == 4:
            raw = struct.pack(f"{bo}{count}I", *(val if isinstance(val, (list, tuple)) else [val]))
        else:
            raise ValueError(f"unsupported type_id {type_id}")

        if len(raw) <= 4:
            field_val = raw.ljust(4, b"\x00")
        else:
            data_pos = extra_data_offset + len(extra_data)
            field_val = struct.pack(f"{bo}I", data_pos)
            extra_data.extend(raw)
            if len(extra_data) % 2 == 1:
                extra_data.append(0)

        ifd_bytes.extend(struct.pack(f"{bo}HHI", tag, type_id, count) + field_val)

    ifd_bytes.extend(struct.pack(f"{bo}I", 0))
    return header + struct.pack(f"{bo}I", ifd_offset) + all_data + bytes(ifd_bytes) + bytes(extra_data)


def test_t4s_scanner_tiff16_rgb_minisblack_and_4ch(tmp_path):
    """T4s: .tif / .tiff with 16-bit RGB, 16-bit MinIsBlack, and 16-bit 4ch RGBI are accepted by
    is_scanner_dng and loaded correctly by load_scanner. 8-bit, 32-bit float, and .dng with photometric=rgb are rejected."""
    # 1. 16-bit 3-channel RGB .tif
    rgb = np.arange(10 * 12 * 3, dtype=np.uint16).reshape(10, 12, 3) * 50
    p_rgb = tmp_path / "neg.tif"
    tifffile.imwrite(p_rgb, rgb, photometric="rgb")
    assert is_scanner_dng(p_rgb)
    fr_rgb = load_scanner(p_rgb)
    assert fr_rgb.rgb.shape == (10, 12, 3)
    np.testing.assert_allclose(fr_rgb.rgb, rgb / 65535.0, atol=1e-6)
    assert "monochrome_dng" not in fr_rgb.meta

    # Also test .tiff suffix
    p_tiff = tmp_path / "neg.tiff"
    tifffile.imwrite(p_tiff, rgb, photometric="rgb")
    assert is_scanner_dng(p_tiff)

    # 2. 16-bit 1-channel MinIsBlack .tif
    mono = np.arange(10 * 12, dtype=np.uint16).reshape(10, 12) * 200
    p_mono = tmp_path / "bw.tif"
    tifffile.imwrite(p_mono, mono, photometric="minisblack")
    assert is_scanner_dng(p_mono)
    fr_mono = load_scanner(p_mono)
    assert fr_mono.meta.get("monochrome_dng") is True
    assert fr_mono.rgb.shape == (10, 12, 3)
    np.testing.assert_allclose(fr_mono.rgb[:, :, 0], mono / 65535.0, atol=1e-6)
    np.testing.assert_array_equal(fr_mono.rgb[:, :, 0], fr_mono.rgb[:, :, 1])

    # 3. 16-bit 4-channel RGBI .tif
    ir = np.full((10, 12), 48000, dtype=np.uint16)
    rgbi = np.dstack([rgb, ir])
    p_rgbi = tmp_path / "rgbi.tif"
    tifffile.imwrite(p_rgbi, rgbi, photometric="rgb")
    assert is_scanner_dng(p_rgbi)
    fr_rgbi = load_scanner(p_rgbi)
    assert fr_rgbi.rgb.shape == (10, 12, 3)
    assert fr_rgbi.meta["infrared_layer"]["source"] == "interleaved_4ch"
    np.testing.assert_allclose(fr_rgbi.rgb, rgb / 65535.0, atol=1e-6)

    # 4. 8-bit .tif is rejected
    rgb8 = (rgb // 256).astype(np.uint8)
    p_8bit = tmp_path / "neg8.tif"
    tifffile.imwrite(p_8bit, rgb8, photometric="rgb")
    assert not is_scanner_dng(p_8bit)
    with pytest.raises(ValueError, match="LinearRaw"):
        load_scanner(p_8bit)

    # 5. 32-bit float .tif is rejected
    p_float = tmp_path / "float32.tif"
    tifffile.imwrite(p_float, (rgb / 65535.0).astype(np.float32), photometric="rgb")
    assert not is_scanner_dng(p_float)
    with pytest.raises(ValueError, match="LinearRaw"):
        load_scanner(p_float)

    # 6. DNG with photometric="rgb" is STILL rejected (DNG LinearRaw contract preserved)
    p_dng_rgb = tmp_path / "rgb_not_linear.dng"
    tifffile.imwrite(p_dng_rgb, rgb, photometric="rgb")
    assert not is_scanner_dng(p_dng_rgb)
    with pytest.raises(ValueError, match="LinearRaw"):
        load_scanner(p_dng_rgb)


def test_t4s_scanner_tiff_lzw_and_packbits_predictor2_decode(tmp_path):
    """T4s: pure-python LZW and PackBits strips with Predictor=2 (Contig and Separate, LE and BE)
    decode 100% identically to original uint16 pixels without imagecodecs."""
    h, w, spp = 16, 24, 3
    rng = np.random.default_rng(2026)
    orig = rng.integers(1000, 60000, (h, w, spp), dtype=np.uint16)
    rows_per_strip = 4
    n_strips = (h + rows_per_strip - 1) // rows_per_strip

    # Case A: LZW + Predictor=2, Contig, Little-Endian
    diff_contig = orig.copy()
    diff_contig[:, 1:, :] = (orig[:, 1:, :] - orig[:, :-1, :]).astype(np.uint16)
    strips_lzw_contig = []
    for s_idx in range(n_strips):
        r0 = s_idx * rows_per_strip
        r1 = min(r0 + rows_per_strip, h)
        chunk = diff_contig[r0:r1].astype("<u2").tobytes()
        strips_lzw_contig.append(encode_tiff_lzw(chunk))

    raw_tiff_lzw = build_synthetic_tiff(strips_lzw_contig, w, h, spp=spp, photometric=2,
                                        compression=5, predictor=2, planarconfig=1,
                                        rows_per_strip=rows_per_strip, byteorder="<")
    p_lzw = tmp_path / "lzw_contig.tif"
    p_lzw.write_bytes(raw_tiff_lzw)

    assert is_scanner_dng(p_lzw)
    fr_lzw = load_scanner(p_lzw)
    np.testing.assert_allclose(fr_lzw.rgb, orig / 65535.0, atol=1e-6)

    # Case B: LZW + Predictor=2, Planar/Separate (planarconfig=2), Little-Endian
    # When planarconfig=2, each plane is diffed across columns independently
    strips_lzw_separate = []
    for c in range(spp):
        plane = orig[:, :, c]
        diff_plane = plane.copy()
        diff_plane[:, 1:] = (plane[:, 1:] - plane[:, :-1]).astype(np.uint16)
        for s_idx in range(n_strips):
            r0 = s_idx * rows_per_strip
            r1 = min(r0 + rows_per_strip, h)
            chunk = diff_plane[r0:r1].astype("<u2").tobytes()
            strips_lzw_separate.append(encode_tiff_lzw(chunk))

    raw_tiff_lzw_sep = build_synthetic_tiff(strips_lzw_separate, w, h, spp=spp, photometric=2,
                                            compression=5, predictor=2, planarconfig=2,
                                            rows_per_strip=rows_per_strip, byteorder="<")
    p_lzw_sep = tmp_path / "lzw_separate.tif"
    p_lzw_sep.write_bytes(raw_tiff_lzw_sep)

    assert is_scanner_dng(p_lzw_sep)
    fr_lzw_sep = load_scanner(p_lzw_sep)
    np.testing.assert_allclose(fr_lzw_sep.rgb, orig / 65535.0, atol=1e-6)

    # Case C: PackBits, Contig, No predictor
    strips_packbits = []
    for s_idx in range(n_strips):
        r0 = s_idx * rows_per_strip
        r1 = min(r0 + rows_per_strip, h)
        chunk = orig[r0:r1].astype("<u2").tobytes()
        strips_packbits.append(encode_packbits(chunk))

    raw_tiff_pb = build_synthetic_tiff(strips_packbits, w, h, spp=spp, photometric=2,
                                       compression=32773, predictor=1, planarconfig=1,
                                       rows_per_strip=rows_per_strip, byteorder="<")
    p_pb = tmp_path / "packbits.tif"
    p_pb.write_bytes(raw_tiff_pb)

    assert is_scanner_dng(p_pb)
    fr_pb = load_scanner(p_pb)
    np.testing.assert_allclose(fr_pb.rgb, orig / 65535.0, atol=1e-6)

    # Case D: LZW, Big-Endian (byteorder=">")
    strips_lzw_be = []
    for s_idx in range(n_strips):
        r0 = s_idx * rows_per_strip
        r1 = min(r0 + rows_per_strip, h)
        chunk = orig[r0:r1].astype(">u2").tobytes()
        strips_lzw_be.append(encode_tiff_lzw(chunk))

    raw_tiff_be = build_synthetic_tiff(strips_lzw_be, w, h, spp=spp, photometric=2,
                                       compression=5, predictor=1, planarconfig=1,
                                       rows_per_strip=rows_per_strip, byteorder=">")
    p_be = tmp_path / "lzw_be.tif"
    p_be.write_bytes(raw_tiff_be)

    assert is_scanner_dng(p_be)
    fr_be = load_scanner(p_be)
    np.testing.assert_allclose(fr_be.rgb, orig / 65535.0, atol=1e-6)


def test_t4s_refuses_filmcolor_exported_positive_tiff(tmp_path):
    """T4s: FilmColor exported positive TIFFs (written by export.write_tiff16 with software='filmcolor'
    and embedded ICC description) are strictly rejected from being treated as scanner negatives."""
    from filmcolor.export import write_tiff16
    pos_data = np.full((30, 40, 3), 0.5, dtype=np.float32)
    p_pos = tmp_path / "exported_positive.tif"
    write_tiff16(p_pos, pos_data, "srgb")

    assert not is_scanner_dng(p_pos)
    with pytest.raises(ValueError, match="LinearRaw"):
        load_scanner(p_pos)


def test_t4s_scanner_truncated_and_unsupported_layout_rejected(tmp_path):
    """T4s: truncated compression streams, corrupt codes, and unsupported layouts are rejected,
    while valid strips that decode to full expected length without EOI (e.g. VueScan output) succeed."""
    from filmcolor.scanner import _decode_tiff_lzw, _decode_packbits
    # 1. Valid LZW strip without trailing EOI and with trailing padding bytes succeeds when expected_len is met
    h, w, spp = 8, 10, 3
    orig = np.full((h, w, spp), 1000, dtype="<u2")
    raw_expected = orig.tobytes()
    expected_bytes = len(raw_expected)
    lzw_no_eoi = encode_tiff_lzw(raw_expected, include_eoi=False) + b"\x00\x00"

    dec_no_eoi = _decode_tiff_lzw(lzw_no_eoi, expected_len=expected_bytes)
    assert dec_no_eoi == raw_expected

    # File-level: load_scanner succeeds on strip without EOI
    raw_tiff_no_eoi = build_synthetic_tiff([lzw_no_eoi], w, h, spp=spp, photometric=2,
                                           compression=5, predictor=1, planarconfig=1)
    p_no_eoi = tmp_path / "valid_no_eoi.tif"
    p_no_eoi.write_bytes(raw_tiff_no_eoi)
    fr_no_eoi = load_scanner(p_no_eoi)
    np.testing.assert_allclose(fr_no_eoi.rgb, orig / 65535.0, atol=1e-6)

    # Missing EOI in standalone decode without expected_len must still be rejected
    with pytest.raises(ValueError, match="missing EOI"):
        _decode_tiff_lzw(lzw_no_eoi, expected_len=None)

    # Truncated LZW strip (under expected length) must be rejected
    bad_lzw = lzw_no_eoi[:len(lzw_no_eoi) // 2]
    raw_bad_lzw = build_synthetic_tiff([bad_lzw], w, h, spp=spp, photometric=2,
                                       compression=5, predictor=1, planarconfig=1)
    p_bad_lzw = tmp_path / "truncated_lzw.tif"
    p_bad_lzw.write_bytes(raw_bad_lzw)

    with pytest.raises(ValueError):
        load_scanner(p_bad_lzw)

    with pytest.raises(ValueError, match="decoded length|truncated"):
        _decode_tiff_lzw(bad_lzw, expected_len=expected_bytes)

    # 2. Corrupt LZW code
    # Construct stream with code >= 258 immediately without table setup
    with pytest.raises(ValueError, match="invalid initial LZW code"):
        # Put 9-bit code 300: (300 >> 1) in byte 0, (300 & 1) << 7 in byte 1
        _decode_tiff_lzw(bytes([150, 0, 0]))

    # 3. Truncated PackBits strip
    good_pb = encode_packbits(orig.tobytes())
    bad_pb = good_pb[:len(good_pb) // 2]
    raw_bad_pb = build_synthetic_tiff([bad_pb], w, h, spp=spp, photometric=2,
                                      compression=32773, predictor=1, planarconfig=1)
    p_bad_pb = tmp_path / "truncated_pb.tif"
    p_bad_pb.write_bytes(raw_bad_pb)

    # load_scanner raises on corrupted/truncated packbits strip
    with pytest.raises(ValueError):
        load_scanner(p_bad_pb)

    # Standalone _decode_packbits raises on truncated stream / length mismatch
    with pytest.raises(ValueError, match="PackBits"):
        _decode_packbits(bad_pb, expected_len=expected_bytes)

    # 4. Unsupported Predictor (e.g. Predictor=3)
    good_lzw = encode_tiff_lzw(raw_expected)
    raw_bad_pred = build_synthetic_tiff([good_lzw], w, h, spp=spp, photometric=2,
                                        compression=5, predictor=3, planarconfig=1)
    p_bad_pred = tmp_path / "bad_predictor.tif"
    p_bad_pred.write_bytes(raw_bad_pred)
    with pytest.raises(ValueError, match="Predictor"):
        load_scanner(p_bad_pred)


def test_t4s_is_scanner_dng_and_load_scanner_frozen_guard(tmp_path, monkeypatch):
    """T4s: verify is_scanner_dng and load_scanner strictly propagate FrozenDataError on frozen inputs,
    and nonexistent files return False for is_scanner_dng without unhandled error."""
    from filmcolor.frozen import FrozenDataError
    # Nonexistent file returns False
    assert not is_scanner_dng(tmp_path / "nonexistent.tif")

    p = tmp_path / "frozen_test.dng"
    p.write_bytes(b"dummy")

    def mock_check_source(path, *, allow_frozen=False, frozen=None):
        if not allow_frozen:
            raise FrozenDataError(f"mock frozen hit: {path}")
        return "mock_sha"

    monkeypatch.setattr("filmcolor.scanner.check_source", mock_check_source)

    with pytest.raises(FrozenDataError):
        is_scanner_dng(p, allow_frozen=False)

    with pytest.raises(FrozenDataError):
        load_scanner(p, allow_frozen=False)


def test_t4s_scanner_large_lzw_strip_bounded_time():
    """T4s: ensure LZW get_code masks bit_buf so multi-megabyte strips decode in linear O(N) time."""
    from filmcolor.scanner import _decode_tiff_lzw
    import time
    raw = (np.arange(100000, dtype=np.uint16) % 256).astype(np.uint8).tobytes()
    enc = encode_tiff_lzw(raw)
    t0 = time.perf_counter()
    dec = _decode_tiff_lzw(enc, len(raw))
    t1 = time.perf_counter()
    assert dec == raw
    assert (t1 - t0) < 0.2, f"LZW decode took {t1 - t0:.4f}s, expected < 0.2s"