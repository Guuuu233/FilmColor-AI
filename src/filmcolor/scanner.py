"""Film-scanner input: linear RGB / RGBI / monochrome DNG (DNG PhotometricInterpretation LinearRaw, 1, 3 or 4 samples), as
written by SilverFast for the Plustek OpticFilm scans of BlueNeg. A second input path next to the
camera RAW path (filmcolor.raw / rawpy_io); both hand the shared inversion a linear, black-subtracted
(H, W, 3) transmission image scaled to the white level, a saturation mask and a film base.

Signal definition (verified on data/raw/public/blueneg samples, see data/manifests/SOURCES.md):
    rgb = (value - BlackLevel) / (WhiteLevel - BlackLevel), per DNG defaults 0 and 2**bits - 1
    when the tags are absent. No CFA, no demosaic, no camera colour matrix. With a LinearizationTable (T4l; VueScan's
    8-bit Plustek DNGs: 256 entries up to WhiteLevel 65535), value = table[stored value] first (DNG specification).
Not verified: whether the scanner software applied infrared dust removal or other processing to
the RGB before writing (its job settings are recorded in the sidecar, not interpreted). The
declared ColorMatrix1 is the standard XYZ->linear sRGB matrix, i.e. a nominal label, not a
calibration. A one-sample infrared layer (transparency mask) or an interleaved fourth infrared
channel is kept out of the colour path; only its statistics are recorded.
"""
from __future__ import annotations

import html
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .base import estimate_base_blocks
from .frozen import check_source

LINEAR_RAW = 34892
PHOTOMETRIC_MINISBLACK = 1
PHOTOMETRIC_RGB = 2
SCANNER_SUFFIXES = (".dng", ".tif", ".tiff")
TIFF_COMPRESSIONS = (1, 5, 8, 32773, 32946)  # Uncompressed, LZW, Deflate (Adobe/PKZIP), PackBits
SILVERFAST_KEYS = ("General_NegativeMode", "isWithIR", "MultiExposure", "General_HdrMode",
                   "IsLayerActive_ISRD", "IsLayerActive_SRDX", "ImageFormat", "General_Bitdepth")


@dataclass
class ScannerFrame:
    rgb: np.ndarray        # (H, W, 3) float32, black-subtracted, / (white - black)
    sat: np.ndarray        # (H, W) bool, any channel at or above white
    black: float
    white: float
    meta: dict


def _decode_packbits(data: bytes, expected_len: int | None = None) -> bytes:
    """Decode TIFF PackBits byte-oriented run-length compressed stream."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        if expected_len is not None and len(out) >= expected_len:
            break
        header = data[i]
        i += 1
        if header < 128:  # 0..127: literal run of (header + 1) bytes
            count = header + 1
            if i + count > n:
                raise ValueError("PackBits stream truncated in literal run")
            out.extend(data[i:i + count])
            i += count
        elif header > 128:  # 129..255 (-127..-1): repeat next byte (257 - header) times
            count = 257 - header
            if i >= n:
                raise ValueError("PackBits stream truncated in repeat run")
            b = data[i]
            i += 1
            out.extend(bytes([b]) * count)
        else:  # header == 128 (0x80): no-op
            pass
    if expected_len is not None and len(out) != expected_len:
        raise ValueError(f"PackBits decoded length {len(out)} != expected {expected_len}")
    return bytes(out)


def _decode_tiff_lzw(data: bytes, expected_len: int | None = None) -> bytes:
    """Decode TIFF LZW stream (MSB-first, 9..12-bit variable code length with early change)."""
    pos = 0
    bit_buf = 0
    bits_in_buf = 0
    n_data = len(data)

    def get_code(size):
        nonlocal pos, bit_buf, bits_in_buf
        while bits_in_buf < size:
            if pos >= n_data:
                return None
            bit_buf = (bit_buf << 8) | data[pos]
            pos += 1
            bits_in_buf += 8
        bits_in_buf -= size
        code = (bit_buf >> bits_in_buf) & ((1 << size) - 1)
        bit_buf &= (1 << bits_in_buf) - 1
        return code

    out = bytearray()
    table = [bytes([i]) for i in range(256)] + [None, None]
    code_size = 9
    old_code = None
    eoi_found = False

    while True:
        if expected_len is not None and len(out) >= expected_len:
            break
        code = get_code(code_size)
        if code is None:
            break
        if code == 257:  # EOI
            eoi_found = True
            break
        if code == 256:  # ClearCode
            table = [bytes([i]) for i in range(256)] + [None, None]
            code_size = 9
            old_code = None
            continue
        if old_code is None:
            if code >= 256:
                raise ValueError(f"invalid initial LZW code {code}")
            entry = table[code]
            out.extend(entry)
            old_code = code
            if expected_len is not None and len(out) > expected_len:
                raise ValueError(f"LZW decoded length {len(out)} exceeds expected {expected_len}")
            continue
        if code < len(table):
            entry = table[code]
        elif code == len(table):
            entry = table[old_code] + table[old_code][:1]
        else:
            raise ValueError(f"corrupt LZW code {code} out of range (table len {len(table)})")
        out.extend(entry)
        if expected_len is not None and len(out) > expected_len:
            raise ValueError(f"LZW decoded length {len(out)} exceeds expected {expected_len}")
        if len(table) < 4096:
            table.append(table[old_code] + entry[:1])
            if len(table) + 1 == (1 << code_size) and code_size < 12:
                code_size += 1
        old_code = code

    if expected_len is not None:
        if len(out) != expected_len:
            raise ValueError(f"LZW decoded length {len(out)} != expected {expected_len}")
    elif not eoi_found:
        raise ValueError("LZW stream truncated (missing EOI code)")
    return bytes(out)


def _read_page_array(page) -> np.ndarray:
    """Read a TIFF page array; fallback to pure Python/NumPy for LZW/PackBits strips if imagecodecs is absent."""
    try:
        arr = page.asarray()
        planarconfig = getattr(page, "planarconfig", 1) or 1
        if planarconfig == 2 and arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        return arr
    except ValueError as exc:
        if "imagecodecs" not in str(exc):
            raise

    # Strictly reject unsupported layouts to prevent silent corruption
    if getattr(page, "is_tiled", False):
        raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: tiled TIFF fallback decoding not supported")
    compression = page.compression
    if compression not in (5, 32773):
        raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: unsupported compression {compression}")
    predictor = int(_tag(page, "Predictor", 1) or 1)
    if predictor not in (1, 2):
        raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: unsupported Predictor {predictor}")
    bits = np.atleast_1d(_tag(page, "BitsPerSample", 0))
    if not (len(bits) > 0 and all(b == 16 for b in bits) and page.dtype.itemsize == 2 and np.dtype(page.dtype).kind == "u"):
        raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: fallback only supports uint16 samples")

    h = page.imagelength
    w = page.imagewidth
    spp = page.samplesperpixel
    rows_per_strip = page.rowsperstrip if page.rowsperstrip is not None else h
    rows_per_strip = max(1, min(rows_per_strip, h))
    offsets = page.dataoffsets
    bytecounts = page.databytecounts
    bo = page.parent.byteorder
    dt = np.dtype(f"{bo}u2")
    planarconfig = getattr(page, "planarconfig", 1) or 1
    fh = page.parent.filehandle

    def _decode_strip(strip_idx: int, rh: int, channels: int) -> np.ndarray:
        expected_bytes = rh * w * channels * 2
        fh.seek(offsets[strip_idx])
        raw = fh.read(bytecounts[strip_idx])
        if len(raw) < bytecounts[strip_idx]:
            raise ValueError(f"truncated strip {strip_idx} in TIFF file")
        dec = _decode_tiff_lzw(raw, expected_bytes) if compression == 5 else _decode_packbits(raw, expected_bytes)
        sarr = np.frombuffer(dec, dtype=dt).astype(page.dtype, copy=False)
        if channels > 1:
            sarr = sarr.reshape(rh, w, channels)
        else:
            sarr = sarr.reshape(rh, w)
        if predictor == 2:
            sarr = np.cumsum(sarr, axis=1, dtype=np.uint16)
        return sarr

    if planarconfig == 1:
        n_strips = (h + rows_per_strip - 1) // rows_per_strip
        if len(offsets) < n_strips or len(bytecounts) < n_strips:
            raise ValueError(f"insufficient strip offsets or bytecounts ({len(offsets)}, {len(bytecounts)} < {n_strips})")
        strips = [_decode_strip(i, min(rows_per_strip, h - i * rows_per_strip), spp) for i in range(n_strips)]
        arr = np.concatenate(strips, axis=0)
    elif planarconfig == 2:
        n_strips_per_plane = (h + rows_per_strip - 1) // rows_per_strip
        total_strips = spp * n_strips_per_plane
        if len(offsets) < total_strips or len(bytecounts) < total_strips:
            raise ValueError(f"insufficient strip offsets or bytecounts for planar TIFF ({len(offsets)}, {len(bytecounts)} < {total_strips})")
        planes = []
        for c in range(spp):
            plane_strips = [
                _decode_strip(c * n_strips_per_plane + j, min(rows_per_strip, h - j * rows_per_strip), 1)
                for j in range(n_strips_per_plane)
            ]
            planes.append(np.concatenate(plane_strips, axis=0))
        arr = np.stack(planes, axis=-1) if spp > 1 else planes[0]
    else:
        raise ValueError(f"unsupported PlanarConfiguration {planarconfig}")

    if arr.dtype != page.dtype:
        arr = arr.astype(page.dtype, copy=False)
    return arr


def _is_filmcolor_export(tf) -> bool:
    """True if this TIFF is a positive export written by FilmColor itself."""
    if not tf.pages:
        return False
    ifd0 = tf.pages[0]
    software = str(_tag(ifd0, "Software", "") or "").strip().lower()
    if software.startswith("filmcolor"):
        return True
    icc = _tag(ifd0, "InterColorProfile")
    if icc is None:
        icc = _tag(ifd0, 34675)
    if icc is not None:
        try:
            raw_icc = bytes(icc)
            if b"FilmColor uncalibrated camera RGB" in raw_icc:
                return True
        except Exception:
            pass
    return False


def _all_pages(tf):
    for p in tf.pages:
        yield p
        for sp in (p.pages or []) if hasattr(p, "pages") else []:
            yield sp


def _tag(page, name, default=None):
    t = page.tags.get(name)
    return default if t is None else t.value


def _main_page(tf, *, allow_tiff: bool = False):
    if not allow_tiff:
        for n_samples in ((3, 4), (1,)):
            cands = [p for p in _all_pages(tf) if p.photometric == LINEAR_RAW and p.samplesperpixel in n_samples
                     and not (_tag(p, "NewSubfileType", 0) or 0) & 1]
            if cands:
                return max(cands, key=lambda p: p.shape[0] * p.shape[1])
        return None

    if _is_filmcolor_export(tf):
        return None

    allowed_photo = (LINEAR_RAW, PHOTOMETRIC_RGB, PHOTOMETRIC_MINISBLACK)
    for n_samples in ((3, 4), (1,)):
        cands = []
        for p in _all_pages(tf):
            if (p.photometric in allowed_photo
                    and p.samplesperpixel in n_samples
                    and p.compression in TIFF_COMPRESSIONS
                    and not (_tag(p, "NewSubfileType", 0) or 0) & 1):
                bits = np.atleast_1d(_tag(p, "BitsPerSample", 0))
                if (len(bits) > 0 and all(b == 16 for b in bits)
                        and getattr(p, "dtype", None) is not None
                        and p.dtype.itemsize == 2 and np.dtype(p.dtype).kind == "u"):
                    cands.append(p)
        if cands:
            return max(cands, key=lambda p: p.shape[0] * p.shape[1])
    return None


def is_scanner_dng(path, *, allow_frozen: bool = False) -> bool:
    """True for a DNG whose main image is 1-, 3- or 4-sample LinearRaw (no CFA),
    or a 16-bit scanner negative TIFF (RGB, RGBI, or MinIsBlack)."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf not in SCANNER_SUFFIXES:
        return False
    try:
        check_source(p, allow_frozen=allow_frozen)
        with tifffile.TiffFile(p) as tf:
            return _main_page(tf, allow_tiff=(suf != ".dng")) is not None
    except (tifffile.TiffFileError, OSError, ValueError):
        return False


def _level(value, n, what):
    if value is None:
        return None
    v = np.atleast_1d(np.asarray(value, np.float64)).ravel()
    if v.size not in (1, n) or not np.all(v == v[0]):
        raise ValueError(f"per-sample or per-pixel {what} not supported: {value}")
    return float(v[0])


def silverfast_settings(xmp) -> dict:
    """Selected SilverFast job settings from its XMP (escaped key/value markup); recorded only."""
    if xmp is None:
        return {}
    s = xmp.decode("utf8", "replace") if isinstance(xmp, bytes) else str(xmp)
    s = urllib.parse.unquote(html.unescape(s))
    out = {}
    for k, v in re.findall(r"<([A-Za-z0-9_]+)>([^<]*)</\1>", s):
        if k in SILVERFAST_KEYS and k not in out:
            out[k] = v.strip()
    return out


def load_scanner(path, *, allow_frozen: bool = False) -> ScannerFrame:
    path = Path(path)
    check_source(path, allow_frozen=allow_frozen)  # sealed dataset frames, also before any hash exists
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    allow_tiff = suf in (".tif", ".tiff")
    with tifffile.TiffFile(path) as tf:
        main = _main_page(tf, allow_tiff=allow_tiff)
        if main is None:
            raise ValueError(f"{path.name}: no 1-, 3- or 4-sample LinearRaw/16-bit scanner image")
        lut_tag = main.tags.get("LinearizationTable")
        lut = None
        if lut_tag is not None:
            lut = np.atleast_1d(np.asarray(lut_tag.value, np.float64)).ravel()
            if lut.size < 2 or not np.all(np.isfinite(lut)) or np.any(np.diff(lut) < 0):
                raise ValueError(f"{path.name}: LinearizationTable must be finite and non-decreasing")
        bits = _tag(main, "BitsPerSample")
        bits = int(np.atleast_1d(bits)[0])
        n_samples = main.samplesperpixel
        black = _level(_tag(main, "BlackLevel"), n_samples, "BlackLevel")
        white = _level(_tag(main, "WhiteLevel"), n_samples, "WhiteLevel")
        black = 0.0 if black is None else black
        white = float(2 ** bits - 1) if white is None else white
        if white <= black:
            raise ValueError(f"{path.name}: white level {white} <= black level {black}")
        a = _read_page_array(main)
        if lut is not None:
            if a.dtype.kind not in "ui":
                raise ValueError(f"{path.name}: LinearizationTable on non-integer samples ({a.dtype})")
            a = lut.astype(np.float32)[np.minimum(a, lut.size - 1)]  # stored values past the table take its last entry
        ifd0 = tf.pages[0]
        ir = [p for p in _all_pages(tf)
              if p is not main and getattr(p, "offset", None) != getattr(main, "offset", -1)
              and p.samplesperpixel == 1 and p.shape[:2] == a.shape[:2]]
        ir_stats = None
        if ir:
            m = _read_page_array(ir[0])[::8, ::8].astype(np.float64)
            ir_stats = dict(new_subfile_type=int(_tag(ir[0], "NewSubfileType", 0) or 0),
                            percentiles={str(q): float(np.percentile(m, q)) for q in (1, 5, 50, 95, 99)},
                            used="no (recorded only)")
        elif a.ndim == 3 and a.shape[2] == 4:
            m = a[::8, ::8, 3].astype(np.float64)
            ir_stats = dict(source="interleaved_4ch",
                            percentiles={str(q): float(np.percentile(m, q)) for q in (1, 5, 50, 95, 99)},
                            used="no (recorded only)")
        meta = dict(make=_tag(ifd0, "Make"), model=_tag(ifd0, "Model"), software=_tag(ifd0, "Software"),
                    unique_camera_model=_tag(ifd0, "UniqueCameraModel"),
                    color_matrix1=list(_tag(ifd0, "ColorMatrix1") or []) or None,
                    as_shot_white_xy=list(_tag(ifd0, "AsShotWhiteXY") or []) or None,
                    bits=bits, black_source="tag" if main.tags.get("BlackLevel") else "DNG default",
                    white_source="tag" if main.tags.get("WhiteLevel") else "DNG default",
                    infrared_layer=ir_stats, scanner_settings=silverfast_settings(_tag(ifd0, "XMP")))
        if n_samples == 1:
            meta["monochrome_dng"] = True
        if lut is not None:  # files without a table keep their previous field set
            meta["linearization_table_len"] = int(lut.size)
    if a.ndim == 2:
        rgb_raw = np.repeat(a[:, :, None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        rgb_raw = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] in (3, 4):
        rgb_raw = a[:, :, :3]
    else:
        raise ValueError(f"{path.name}: unexpected image shape {a.shape}")
    sat_ch = rgb_raw >= white
    if (allow_tiff and n_samples >= 3 and float(sat_ch[:, :, 0].mean()) > 0.30
            and float(sat_ch[:, :, 1].mean()) < 0.05 and float(sat_ch[:, :, 2].mean()) < 0.05):
        # T4u: a pre-stretched orange mask can clip R while G and B retain usable signal
        sat = sat_ch[:, :, 1] | sat_ch[:, :, 2]
        meta["single_channel_prestretched"] = "R"
    else:
        sat = np.any(sat_ch, -1)
    del sat_ch
    rgb = ((rgb_raw.astype(np.float32) - np.float32(black)) / np.float32(white - black))
    del a, rgb_raw
    if not np.all(np.isfinite(rgb)):
        raise ValueError(f"{path.name}: non-finite samples")
    return ScannerFrame(rgb=rgb, sat=sat, black=black, white=white, meta=meta)


def block_means(rgb, sat, block: int):
    """(hb, wb, 3) block means and (hb, wb) any-saturated flags; partial edge blocks dropped."""
    hb, wb = rgb.shape[0] // block, rgb.shape[1] // block
    if hb == 0 or wb == 0:
        raise ValueError(f"image {rgb.shape[:2]} smaller than one {block}px block")
    r = rgb[: hb * block, : wb * block].reshape(hb, block, wb, block, 3)
    s = sat[: hb * block, : wb * block].reshape(hb, block, wb, block)
    return r.mean((1, 3), dtype=np.float64).astype(np.float32), s.any((1, 3))


def estimate_base_scanner(frame: ScannerFrame, *, block: int = 24, **kwargs):
    """Automatic film base on the scanner image (same block selection as the camera path)."""
    sp, sp_sat = block_means(frame.rgb, frame.sat, block)
    return estimate_base_blocks(sp, sp_sat, block=block, **kwargs)
