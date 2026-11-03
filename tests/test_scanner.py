"""Scanner input path (linear RGB DNG) on synthetic files."""
import struct
import zlib
import numpy as np
import pytest
import tifffile

from filmcolor.pipeline import convert
from filmcolor.scanner import block_means, estimate_base_scanner, is_scanner_dng, load_scanner


def srgb_eotf_ref(e):
    e = np.clip(e, 0.0, 1.0)
    return np.where(e <= 0.04045, e / 12.92, ((e + 0.055) / 1.055) ** 2.4)

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
                         byteorder: str = "<", bits_per_sample: int = 16,
                         extra_entries: list | None = None) -> bytes:
    """Build a valid standalone TIFF in memory with custom strip data and compression."""
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
        (258, 3, spp, [bits_per_sample] * spp),
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
    if extra_entries:
        entries.extend(extra_entries)

    entries.sort(key=lambda x: x[0])
    ifd_len = 2 + len(entries) * 12 + 4
    extra_data_offset = ifd_offset + ifd_len

    ifd_bytes = bytearray(struct.pack(f"{bo}H", len(entries)))
    extra_data = bytearray()

    for tag, type_id, count, val in entries:
        if type_id == 1:
            raw = bytes(val)
        elif type_id == 2:
            raw = (val.encode("ascii") + b"\x00") if isinstance(val, str) else bytes(val)
            count = len(raw)
        elif type_id == 3:
            raw = struct.pack(f"{bo}{count}H", *(val if isinstance(val, (list, tuple)) else [val]))
        elif type_id == 4:
            raw = struct.pack(f"{bo}{count}I", *(val if isinstance(val, (list, tuple)) else [val]))
        elif type_id == 7:
            raw = bytes(val)
            count = len(raw)
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


def build_synthetic_icc(dev_class: bytes = b"mntr", sig: bytes = b"rTRC",
                        trc_type: str = "curv", gamma: float | None = None,
                        lut: np.ndarray | None = None, ftype: int = 3,
                        corrupt: str | None = None) -> bytes:
    """Build a minimal valid or intentionally corrupted ICC profile for testing TRC parsing."""
    if corrupt == "truncated":
        return b"short_icc"
    hdr = bytearray(128)
    hdr[12:16] = dev_class

    t_data = bytearray()
    if corrupt == "unknown_type":
        t_data.extend(b"xxxx\x00\x00\x00\x00\x00\x00\x00\x00")
    elif trc_type == "curv":
        t_data.extend(b"curv\x00\x00\x00\x00")
        if gamma is not None:
            t_data.extend(struct.pack(">I", 1))
            t_data.extend(struct.pack(">H", max(0, min(65535, int(round(gamma * 256.0))))))
        elif lut is not None:
            cnt = len(lut)
            t_data.extend(struct.pack(">I", cnt))
            t_data.extend((np.clip(lut, 0.0, 1.0) * 65535.0).astype(">u2").tobytes())
        else:
            t_data.extend(struct.pack(">I", 0))
    elif trc_type == "para":
        t_data.extend(b"para\x00\x00\x00\x00")
        t_data.extend(struct.pack(">H", ftype))
        t_data.extend(b"\x00\x00")
        gam_val = int(round((gamma if gamma is not None else 2.2) * 65536.0))
        t_data.extend(struct.pack(">i", gam_val))

    tag_count = 1
    t_off = 132 + 12
    t_sz = len(t_data)
    if corrupt == "offset_overflow":
        t_off = 999999

    tag_entry = struct.pack(">4sII", sig, t_off, t_sz)
    total_len = 128 + 4 + 12 + t_sz
    struct.pack_into(">I", hdr, 0, total_len)

    icc = bytes(hdr + struct.pack(">I", tag_count) + tag_entry + t_data)
    if corrupt == "tag_count_overflow":
        icc = bytearray(icc)
        struct.pack_into(">I", icc, 128, 99999)
        icc = bytes(icc)
    return icc


def test_t4s_scanner_tiff16_rgb_minisblack_and_4ch(tmp_path):
    """T4s/T4v: .tif / .tiff with 8/16-bit RGB, 8/16-bit MinIsBlack, and 16-bit 4ch RGBI are accepted by
    is_scanner_dng and loaded correctly by load_scanner. Non-standard bits, 32-bit float, and .dng with photometric=rgb are rejected."""
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

    # 4. 8-bit .tif is accepted and linearised via sRGB EOTF
    rgb8 = (rgb // 256).astype(np.uint8)
    p_8bit = tmp_path / "neg8.tif"
    tifffile.imwrite(p_8bit, rgb8, photometric="rgb")
    assert is_scanner_dng(p_8bit)
    fr8 = load_scanner(p_8bit)
    assert fr8.rgb.shape == (10, 12, 3)
    assert fr8.meta["bits"] == 8
    assert fr8.meta["tiff_eotf"] == "srgb"
    np.testing.assert_allclose(fr8.rgb, srgb_eotf_ref(rgb8 / 255.0), atol=1e-5)

    # 4b. 12-bit / non-standard bit depth .tif is rejected
    raw_12bit = build_synthetic_tiff([rgb.astype("<u2").tobytes()], 12, 10, spp=3, photometric=2,
                                     bits_per_sample=12)
    p_12bit = tmp_path / "neg12.tif"
    p_12bit.write_bytes(raw_12bit)
    assert not is_scanner_dng(p_12bit)
    with pytest.raises(ValueError, match="LinearRaw|scanner image"):
        load_scanner(p_12bit)

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


def test_t4u_scanner_tiff_single_channel_prestretched_orange_mask_sat(tmp_path):
    from filmcolor.scanner import estimate_base_scanner, is_scanner_dng, load_scanner

    h, w = 96, 96
    rgb16 = np.zeros((h, w, 3), dtype=np.uint16)
    rgb16[:, :] = [55000, 43000, 36000]

    # R channel: 80% saturated (65535), remaining 20% normal
    n_sat = int(0.80 * h * w)
    rgb16.reshape(-1, 3)[:n_sat, 0] = 65535

    p_tiff = tmp_path / "orange_prestretched.tif"
    tifffile.imwrite(p_tiff, rgb16, photometric="rgb")
    assert is_scanner_dng(p_tiff)

    fr = load_scanner(p_tiff)
    assert fr.meta.get("single_channel_prestretched") == "R"
    assert fr.sat.mean() == 0.0
    base = estimate_base_scanner(fr, block=24)
    assert base is not None
    assert len(base.rgb) == 3


def test_t4u_scanner_tiff_true_multi_channel_saturation_still_masked(tmp_path):
    from filmcolor.scanner import is_scanner_dng, load_scanner

    h, w = 48, 48
    rgb16 = np.zeros((h, w, 3), dtype=np.uint16)
    rgb16[:, :, 0] = 50000
    rgb16[:, :, 1] = 40000
    rgb16[:, :, 2] = 30000

    # > 10% saturated in both R and G (e.g. 15% = ~345 pixels)
    n_sat = int(0.15 * h * w)
    rgb16.reshape(-1, 3)[:n_sat, 0] = 65535
    rgb16.reshape(-1, 3)[:n_sat, 1] = 65535

    p_multi = tmp_path / "multi_sat.tif"
    tifffile.imwrite(p_multi, rgb16, photometric="rgb")
    assert is_scanner_dng(p_multi)

    fr = load_scanner(p_multi)
    assert "single_channel_prestretched" not in fr.meta
    expected_sat = np.any(rgb16 >= 65535, -1)
    np.testing.assert_array_equal(fr.sat, expected_sat)
    assert fr.sat.mean() == pytest.approx(0.15, abs=1e-3)

    # Also verify .dng single-channel saturation does NOT trigger single_channel_prestretched
    p_dng = tmp_path / "single_sat.dng"
    dng_raw = np.full((h, w, 3), 30000, dtype=np.uint16)
    dng_raw.reshape(-1, 3)[:int(0.50 * h * w), 0] = 65535
    with tifffile.TiffWriter(p_dng) as tw:
        tw.write(dng_raw, photometric=34892, planarconfig="contig")
    assert is_scanner_dng(p_dng)
    fr_dng = load_scanner(p_dng)
    assert "single_channel_prestretched" not in fr_dng.meta
    np.testing.assert_array_equal(fr_dng.sat, np.any(dng_raw >= 65535, -1))


def test_t4v_scanner_tiff8_lzw_deflate_packbits_and_predictor2_fallback(tmp_path, monkeypatch):
    """T4v: pure-Python fallback decodes 8-bit 1ch/3ch/4ch strips under LZW (5), Deflate (8 and 32946),
    and PackBits (32773) with Predictor=2, and rejects truncated/corrupt Deflate streams."""
    def mock_asarray(self, *args, **kwargs):
        raise ValueError("codec requires the 'imagecodecs' package")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", mock_asarray)

    # 1. 8-bit 1-channel MinIsBlack (spp=1), Deflate (compression=8), Predictor=2
    h, w = 12, 16
    rng = np.random.default_rng(2026)
    orig_mono = rng.integers(10, 240, (h, w), dtype=np.uint8)
    diff_mono = orig_mono.copy()
    diff_mono[:, 1:] = (orig_mono[:, 1:] - orig_mono[:, :-1]).astype(np.uint8)
    strip_def8 = zlib.compress(diff_mono.tobytes())
    raw_mono = build_synthetic_tiff([strip_def8], w, h, spp=1, photometric=1, compression=8,
                                    predictor=2, bits_per_sample=8)
    p_mono = tmp_path / "mono8_def8.tif"
    p_mono.write_bytes(raw_mono)
    assert is_scanner_dng(p_mono)
    fr_mono = load_scanner(p_mono)
    assert fr_mono.rgb.shape == (h, w, 3)
    assert fr_mono.meta["bits"] == 8
    assert fr_mono.meta["monochrome_dng"] is True
    assert fr_mono.meta["tiff_eotf"] == "srgb"
    np.testing.assert_allclose(fr_mono.rgb[:, :, 0], srgb_eotf_ref(orig_mono / 255.0), atol=1e-5)
    np.testing.assert_array_equal(fr_mono.rgb[:, :, 0], fr_mono.rgb[:, :, 1])

    # 2. 8-bit 3-channel RGB (spp=3), Deflate (compression=32946), Predictor=2
    orig_rgb = rng.integers(10, 240, (h, w, 3), dtype=np.uint8)
    diff_rgb = orig_rgb.copy()
    diff_rgb[:, 1:, :] = (orig_rgb[:, 1:, :] - orig_rgb[:, :-1, :]).astype(np.uint8)
    strip_def32946 = zlib.compress(diff_rgb.tobytes())
    raw_rgb = build_synthetic_tiff([strip_def32946], w, h, spp=3, photometric=2, compression=32946,
                                   predictor=2, bits_per_sample=8)
    p_rgb = tmp_path / "rgb8_def32946.tif"
    p_rgb.write_bytes(raw_rgb)
    assert is_scanner_dng(p_rgb)
    fr_rgb = load_scanner(p_rgb)
    assert fr_rgb.rgb.shape == (h, w, 3)
    assert fr_rgb.meta["bits"] == 8
    assert fr_rgb.meta["tiff_eotf"] == "srgb"
    np.testing.assert_allclose(fr_rgb.rgb, srgb_eotf_ref(orig_rgb / 255.0), atol=1e-5)

    # 3. 8-bit 4-channel RGBI (spp=4), LZW (compression=5), Predictor=2
    orig_rgbi = rng.integers(10, 240, (h, w, 4), dtype=np.uint8)
    diff_rgbi = orig_rgbi.copy()
    diff_rgbi[:, 1:, :] = (orig_rgbi[:, 1:, :] - orig_rgbi[:, :-1, :]).astype(np.uint8)
    strip_lzw = encode_tiff_lzw(diff_rgbi.tobytes())
    raw_rgbi = build_synthetic_tiff([strip_lzw], w, h, spp=4, photometric=2, compression=5,
                                    predictor=2, bits_per_sample=8)
    p_rgbi = tmp_path / "rgbi8_lzw.tif"
    p_rgbi.write_bytes(raw_rgbi)
    assert is_scanner_dng(p_rgbi)
    fr_rgbi = load_scanner(p_rgbi)
    assert fr_rgbi.rgb.shape == (h, w, 3)
    assert fr_rgbi.meta["infrared_layer"]["source"] == "interleaved_4ch"
    np.testing.assert_allclose(fr_rgbi.rgb, srgb_eotf_ref(orig_rgbi[:, :, :3] / 255.0), atol=1e-5)

    # 4. 8-bit 3-channel RGB (spp=3), PackBits (compression=32773), Predictor=2
    strip_pb = encode_packbits(diff_rgb.tobytes())
    raw_pb = build_synthetic_tiff([strip_pb], w, h, spp=3, photometric=2, compression=32773,
                                  predictor=2, bits_per_sample=8)
    p_pb = tmp_path / "rgb8_packbits.tif"
    p_pb.write_bytes(raw_pb)
    assert is_scanner_dng(p_pb)
    fr_pb = load_scanner(p_pb)
    np.testing.assert_allclose(fr_pb.rgb, srgb_eotf_ref(orig_rgb / 255.0), atol=1e-5)

    # 5. Corrupt / truncated Deflate stream is rejected with ValueError
    corrupt_strip = b"\x78\x9c\xff\xff\x00"
    raw_corrupt = build_synthetic_tiff([corrupt_strip], w, h, spp=3, photometric=2, compression=8,
                                       predictor=1, bits_per_sample=8)
    p_corrupt = tmp_path / "corrupt_deflate.tif"
    p_corrupt.write_bytes(raw_corrupt)
    with pytest.raises(ValueError, match="corrupt Deflate strip|imagecodecs"):
        load_scanner(p_corrupt)


def test_t4v_scanner_tiff16_mntr_icc_gamma_and_lut_eotf_applied_scnr_ignored(tmp_path):
    """T4v: 16-bit TIFFs with display/space ICC (mntr/spac) apply gamma or LUT EOTF and record tiff_eotf,
    while scanner profile (scnr) preserves raw linear values; malformed tags fall back safely."""
    h, w = 8, 12
    orig = np.linspace(1000, 64000, h * w * 3, dtype=np.uint16).reshape(h, w, 3)
    data = orig.astype("<u2").tobytes()

    # 1. mntr class ICC with single gamma (e.g. 1.80 ROMM RGB curv)
    icc_gamma = build_synthetic_icc(dev_class=b"mntr", sig=b"rTRC", trc_type="curv", gamma=1.80)
    raw_gamma = build_synthetic_tiff([data], w, h, spp=3, photometric=2, bits_per_sample=16,
                                     extra_entries=[(34675, 7, len(icc_gamma), icc_gamma)])
    p_gamma = tmp_path / "mntr_gamma18.tif"
    p_gamma.write_bytes(raw_gamma)
    assert is_scanner_dng(p_gamma)
    fr_gamma = load_scanner(p_gamma)
    assert fr_gamma.meta["tiff_eotf"] == f"gamma:{461.0 / 256.0:.4f}"
    expected_gamma = np.power(orig / 65535.0, np.float32(461.0 / 256.0))
    np.testing.assert_allclose(fr_gamma.rgb, expected_gamma, atol=1e-5)

    # 2. mntr class ICC with 256-point LUT
    lut = (np.linspace(0.0, 1.0, 256) ** 2.2).astype(np.float32)
    icc_lut = build_synthetic_icc(dev_class=b"mntr", sig=b"gTRC", trc_type="curv", lut=lut)
    raw_lut = build_synthetic_tiff([data], w, h, spp=3, photometric=2, bits_per_sample=16,
                                   extra_entries=[(34675, 7, len(icc_lut), icc_lut)])
    p_lut = tmp_path / "mntr_lut.tif"
    p_lut.write_bytes(raw_lut)
    assert is_scanner_dng(p_lut)
    fr_lut = load_scanner(p_lut)
    assert fr_lut.meta["tiff_eotf"] == "icc_lut:256"
    # The stored 16-bit LUT is quantized to uint16 / 65535.0
    lut_quantized = ((np.clip(lut, 0.0, 1.0) * 65535.0).astype(">u2")).astype(np.float32) / 65535.0
    expected_lut = np.interp(orig / 65535.0, np.linspace(0.0, 1.0, 256), lut_quantized)
    np.testing.assert_allclose(fr_lut.rgb, expected_lut, atol=1e-6)

    # 3. scnr class ICC with gamma 2.2 (scanner input device profile: must be kept linear)
    icc_scnr = build_synthetic_icc(dev_class=b"scnr", sig=b"rTRC", trc_type="curv", gamma=2.20)
    raw_scnr = build_synthetic_tiff([data], w, h, spp=3, photometric=2, bits_per_sample=16,
                                    extra_entries=[(34675, 7, len(icc_scnr), icc_scnr)])
    p_scnr = tmp_path / "scnr_gamma.tif"
    p_scnr.write_bytes(raw_scnr)
    assert is_scanner_dng(p_scnr)
    fr_scnr = load_scanner(p_scnr)
    assert "tiff_eotf" not in fr_scnr.meta
    np.testing.assert_allclose(fr_scnr.rgb, orig / 65535.0, atol=1e-6)

    # 4. Boundary & corrupt ICC tags default safely to linear (16-bit) without crash
    for err_type in ("truncated", "tag_count_overflow", "offset_overflow", "unknown_type"):
        bad_icc = build_synthetic_icc(dev_class=b"mntr", corrupt=err_type)
        if err_type == "unknown_type":
            assert bad_icc[144:148] == b"xxxx"
        raw_bad = build_synthetic_tiff([data], w, h, spp=3, photometric=2, bits_per_sample=16,
                                       extra_entries=[(34675, 7, len(bad_icc), bad_icc)])
        p_bad = tmp_path / f"bad_{err_type}.tif"
        p_bad.write_bytes(raw_bad)
        assert is_scanner_dng(p_bad)
        fr_bad = load_scanner(p_bad)
        assert "tiff_eotf" not in fr_bad.meta
        np.testing.assert_allclose(fr_bad.rgb, orig / 65535.0, atol=1e-6)


def test_t4v_prestretched_tiff_pedestal_and_r_tail_reconstruction(tmp_path):
    """T4v: 16-bit negative TIFF with pre-stretched R saturation (>50% saturated, r_p01>0.45)
    and zero-crushed blue pedestal (b_p01<0.03) triggers reconstruction, smooth R tail, and avoids channel cap."""
    from filmcolor.invert import invert_negative

    h, w = 96, 128
    rgb16 = np.zeros((h, w, 3), dtype=np.uint16)
    # Left 48 columns: unexposed film base area
    rgb16[:, :48] = [65535, 42000, 30000]

    # Right 80 columns: negative scene
    scene_w = 80
    rgb16[:, 48:, 1] = np.linspace(4000, 40000, h * scene_w).reshape(h, scene_w).astype(np.uint16)
    b_vals = np.linspace(0, 30000, h * scene_w)
    b_vals[:int(0.10 * h * scene_w)] = 0  # bottom 10% crushed to 0
    rgb16[:, 48:, 2] = b_vals.reshape(h, scene_w).astype(np.uint16)

    r_vals = np.full(h * scene_w, 65535, dtype=np.uint16)
    unclip_count = int(0.40 * h * scene_w)
    r_vals[:unclip_count] = np.linspace(32000, 64000, unclip_count).astype(np.uint16)
    rgb16[:, 48:, 0] = r_vals.reshape(h, scene_w)

    p = tmp_path / "prestretched_neg.tif"
    tifffile.imwrite(p, rgb16, photometric="rgb")
    assert is_scanner_dng(p)

    fr = load_scanner(p)
    assert fr.meta.get("single_channel_prestretched") == "R"
    assert "prestretched_reconstruction" in fr.meta
    rinfo = fr.meta["prestretched_reconstruction"]
    assert rinfo["gamma_eotf"] == 2.2
    assert rinfo["pedestal_p0"] > 0
    assert rinfo["r_base_est"] >= 1.0
    assert 0 < fr.rgb.min() < fr.rgb.max() <= 1.0

    # Invert and verify white point channel_cap is avoided
    base = estimate_base_scanner(fr, block=16)
    res = invert_negative(fr.rgb, base.rgb, white_block=16)
    assert "channel_cap" not in res.report.get("white_point", {})

    from filmcolor.scanner import _reconstruct_prestretched_rgb

    transition = np.empty((16, 1001, 3), dtype=np.float32)
    transition[..., 0] = np.linspace(0.989, 1.0, 1001, dtype=np.float32)
    transition[..., 1] = 0.6
    transition[..., 2] = 0.0
    before = transition.copy()
    reconstructed, info = _reconstruct_prestretched_rgb(transition)
    assert info is not None
    steps = np.diff(reconstructed[0, :, 0])
    assert np.all(steps >= 0.0)
    assert float(steps.max()) < 1e-3
    np.testing.assert_array_equal(transition, before)


def test_t4v_prestretched_tiff_without_zero_crushed_blue_skips_reconstruction(tmp_path):
    """T4v: single-channel R saturation without zero-crushed blue (b_p01 >= 0.03) or with mntr ICC
    only decouples saturation mask and skips prestretched_reconstruction."""
    h, w = 96, 96
    rgb16 = np.zeros((h, w, 3), dtype=np.uint16)
    rgb16[:, :] = [55000, 43000, 36000]
    n_sat = int(0.70 * h * w)
    rgb16.reshape(-1, 3)[:n_sat, 0] = 65535

    # Case A: b_p01 is 36000/65535 = 0.549 >= 0.03
    p_uncrushed = tmp_path / "uncrushed_r_sat.tif"
    tifffile.imwrite(p_uncrushed, rgb16, photometric="rgb")
    fr_a = load_scanner(p_uncrushed)
    assert fr_a.meta.get("single_channel_prestretched") == "R"
    assert fr_a.sat.mean() == 0.0
    assert "prestretched_reconstruction" not in fr_a.meta

    # Case B: zero-crushed blue, BUT has mntr ICC (mode != 'linear')
    rgb16_crushed = rgb16.copy()
    rgb16_crushed.reshape(-1, 3)[:int(0.10 * h * w), 2] = 0
    icc = build_synthetic_icc(dev_class=b"mntr", gamma=1.8)
    data = rgb16_crushed.astype("<u2").tobytes()
    raw_icc = build_synthetic_tiff([data], w, h, spp=3, photometric=2, bits_per_sample=16,
                                   extra_entries=[(34675, 7, len(icc), icc)])
    p_icc = tmp_path / "crushed_with_icc.tif"
    p_icc.write_bytes(raw_icc)
    fr_b = load_scanner(p_icc)
    assert fr_b.meta.get("single_channel_prestretched") == "R"
    assert "prestretched_reconstruction" not in fr_b.meta
    assert fr_b.meta.get("tiff_eotf") == f"gamma:{461.0 / 256.0:.4f}"


def test_t4v_refuses_8bit_filmcolor_export_and_unsupported_bit_depths(tmp_path):
    """T4v: FilmColor 8-bit positive exports and unsupported bit depths (4-bit, 32-bit) are rejected."""
    # 1. 8-bit positive export with Software="FilmColor positive export"
    data8 = np.full((16, 16, 3), 128, dtype=np.uint8)
    raw_fc_soft = build_synthetic_tiff([data8.tobytes()], 16, 16, spp=3, photometric=2,
                                       bits_per_sample=8, extra_entries=[(305, 2, 0, "FilmColor positive export")])
    p_fc_soft = tmp_path / "fc8_software.tif"
    p_fc_soft.write_bytes(raw_fc_soft)
    assert not is_scanner_dng(p_fc_soft)
    with pytest.raises(ValueError, match="LinearRaw|scanner image"):
        load_scanner(p_fc_soft)

    # 2. 8-bit positive export with FilmColor ICC description
    fc_desc_icc = b"\x00" * 128 + b"\x00" * 4 + b"FilmColor uncalibrated camera RGB"
    raw_fc_icc = build_synthetic_tiff([data8.tobytes()], 16, 16, spp=3, photometric=2,
                                      bits_per_sample=8, extra_entries=[(34675, 7, len(fc_desc_icc), fc_desc_icc)])
    p_fc_icc = tmp_path / "fc8_icc.tif"
    p_fc_icc.write_bytes(raw_fc_icc)
    assert not is_scanner_dng(p_fc_icc)
    with pytest.raises(ValueError, match="LinearRaw|scanner image"):
        load_scanner(p_fc_icc)

    # 3. 4-bit TIFF is rejected
    raw_4bit = build_synthetic_tiff([data8.tobytes()[:16 * 8]], 16, 16, spp=1, photometric=1,
                                    bits_per_sample=4)
    p_4bit = tmp_path / "unsupported_4bit.tif"
    p_4bit.write_bytes(raw_4bit)
    assert not is_scanner_dng(p_4bit)
    with pytest.raises(ValueError, match="LinearRaw|scanner image"):
        load_scanner(p_4bit)

    # 4. 32-bit TIFF is rejected
    data32 = np.full((16, 16, 3), 50000, dtype=np.uint32)
    p_32bit = tmp_path / "unsupported_32bit.tif"
    tifffile.imwrite(p_32bit, data32, photometric="rgb")
    assert not is_scanner_dng(p_32bit)
    with pytest.raises(ValueError, match="LinearRaw|scanner image"):
        load_scanner(p_32bit)
