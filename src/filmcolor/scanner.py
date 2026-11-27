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
import struct
import urllib.parse
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .base import estimate_base_blocks
from .frozen import check_source
from .invert import _all_finite
from .render import srgb_eotf as _srgb_eotf

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


_LZW_BASE_TABLE: list[bytes | None] = [bytes([i]) for i in range(256)] + [None, None]


def _decode_tiff_lzw(data: bytes, expected_len: int | None = None) -> bytes:
    """Decode TIFF LZW stream (MSB-first, 9..12-bit variable code length with early change)."""
    pos = 0
    bit_buf = 0
    bits_in_buf = 0
    n_data = len(data)
    n_data_m4 = n_data - 4

    out = bytearray()
    out_extend = out.extend
    out_len = 0
    has_expected = expected_len is not None
    table = _LZW_BASE_TABLE.copy()
    table_append = table.append
    table_len = 258
    code_size = 9
    code_mask = 511
    next_bump = 511
    old_code = -1
    eoi_found = False

    while True:
        if has_expected and out_len >= expected_len:
            break
        if bits_in_buf < code_size:
            bit_buf &= (1 << bits_in_buf) - 1
            if pos <= n_data_m4:
                bit_buf = (
                    (bit_buf << 32)
                    | (data[pos] << 24)
                    | (data[pos + 1] << 16)
                    | (data[pos + 2] << 8)
                    | data[pos + 3]
                )
                pos += 4
                bits_in_buf += 32
            else:
                while bits_in_buf < code_size:
                    if pos >= n_data:
                        break
                    bit_buf = (bit_buf << 8) | data[pos]
                    pos += 1
                    bits_in_buf += 8
                if bits_in_buf < code_size:
                    break
        bits_in_buf -= code_size
        code = (bit_buf >> bits_in_buf) & code_mask

        if code == 257:  # EOI
            eoi_found = True
            break
        if code == 256:  # ClearCode
            table = _LZW_BASE_TABLE.copy()
            table_append = table.append
            table_len = 258
            code_size = 9
            code_mask = 511
            next_bump = 511
            old_code = -1
            continue
        if old_code < 0:
            if code >= 256:
                raise ValueError(f"invalid initial LZW code {code}")
            entry = table[code]
            out_extend(entry)
            out_len += len(entry)
            old_code = code
            if has_expected and out_len > expected_len:
                raise ValueError(f"LZW decoded length {out_len} exceeds expected {expected_len}")
            continue
        if code < table_len:
            entry = table[code]
        elif code == table_len:
            prev = table[old_code]
            entry = prev + prev[:1]
        else:
            raise ValueError(f"corrupt LZW code {code} out of range (table len {table_len})")
        out_extend(entry)
        out_len += len(entry)
        if has_expected and out_len > expected_len:
            raise ValueError(f"LZW decoded length {out_len} exceeds expected {expected_len}")
        if table_len < 4096:
            table_append(table[old_code] + entry[:1])
            table_len += 1
            if table_len == next_bump and code_size < 12:
                code_size += 1
                code_mask = (1 << code_size) - 1
                next_bump = code_mask
        old_code = code

    if has_expected:
        if out_len != expected_len:
            raise ValueError(f"LZW decoded length {out_len} != expected {expected_len}")
    elif not eoi_found:
        raise ValueError("LZW stream truncated (missing EOI code)")
    return bytes(out)


def _read_page_array(page) -> np.ndarray:
    """Read a TIFF page array; fallback to pure Python/NumPy for LZW/PackBits strips if imagecodecs is absent."""
    predictor = int(_tag(page, "Predictor", 1) or 1)
    if predictor not in (1, 2) and not (predictor == 3 and np.dtype(page.dtype).kind == "f"):
        raise ValueError(f"unsupported TIFF Predictor={predictor} for dtype {page.dtype}")
    try:
        arr = page.asarray()
        planarconfig = getattr(page, "planarconfig", 1) or 1
        if planarconfig == 2 and arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        return arr
    except (ValueError, NotImplementedError) as exc:
        if "imagecodecs" not in str(exc):
            raise
    except Exception as exc:
        if type(exc).__module__.startswith("imagecodecs"):
            raise ValueError(f"corrupt compressed TIFF segment: {exc}") from exc
        raise

    compression = page.compression
    predictor = int(_tag(page, "Predictor", 1) or 1)
    itemsize = page.dtype.itemsize
    if (getattr(page, "is_tiled", False) and compression in (8, 32946)
            and predictor == 3 and np.dtype(page.dtype).kind == "f" and itemsize in (2, 4)):
        planarconfig = getattr(page, "planarconfig", 1) or 1
        if planarconfig != 1:
            raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: PlanarConfiguration={planarconfig} tiled float fallback not supported")
        h, w, spp = page.imagelength, page.imagewidth, page.samplesperpixel
        tw, tl = page.tilewidth, page.tilelength
        nt_y, nt_x = (h + tl - 1) // tl, (w + tw - 1) // tw
        n_tiles = nt_y * nt_x
        if len(page.dataoffsets) < n_tiles or len(page.databytecounts) < n_tiles:
            raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: truncated TIFF tile table ({len(page.dataoffsets)}/{len(page.databytecounts)} < {n_tiles})")
        expected_dec = tl * tw * spp * itemsize
        out = np.empty((nt_y * tl, nt_x * tw, spp) if spp > 1 else (nt_y * tl, nt_x * tw), dtype=page.dtype)
        fh = page.parent.filehandle
        fdt = ">f2" if itemsize == 2 else ">f4"
        for idx in range(n_tiles):
            off, bc = page.dataoffsets[idx], page.databytecounts[idx]
            ty, tx = divmod(idx, nt_x)
            fh.seek(off)
            raw = fh.read(bc)
            if len(raw) != bc:
                raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: truncated TIFF tile {idx} ({len(raw)} != {bc})")
            try:
                dec = zlib.decompress(raw)
            except zlib.error as exc:
                raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: corrupt deflate TIFF tile {idx}: {exc}") from exc
            if len(dec) != expected_dec:
                raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: decoded tile {idx} length {len(dec)} != expected {expected_dec}")
            r = np.cumsum(np.frombuffer(dec, dtype=np.uint8).reshape(tl, -1, spp), axis=1, dtype=np.uint8)
            be = np.transpose(r.reshape(tl, itemsize, tw * spp), (0, 2, 1)).copy()
            tile = be.view(fdt).reshape((tl, tw, spp) if spp > 1 else (tl, tw)).astype(page.dtype, copy=False)
            out[ty * tl:(ty + 1) * tl, tx * tw:(tx + 1) * tw] = tile
        return out[:h, :w]
    if getattr(page, "photometric", None) == LINEAR_RAW:
        fpath = getattr(getattr(page.parent, "filehandle", None), "path", None)
        if fpath:
            import rawpy
            with rawpy.imread(str(fpath)) as rp:
                raw_arr = rp.raw_image.copy()
            h, w, spp = page.imagelength, page.imagewidth, page.samplesperpixel
            if spp == 1 and raw_arr.ndim == 2 and raw_arr.shape == (h, w):
                return raw_arr
            if spp in (3, 4) and raw_arr.ndim == 3 and raw_arr.shape[:2] == (h, w) and raw_arr.shape[2] >= spp:
                return raw_arr[:, :, :spp]

    # Strictly reject unsupported layouts to prevent silent corruption
    if getattr(page, "is_tiled", False):
        raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: tiled TIFF fallback decoding not supported")
    compression = page.compression
    if compression not in (5, 8, 32773, 32946):
        raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: unsupported compression {compression}")
    predictor = int(_tag(page, "Predictor", 1) or 1)
    if predictor not in (1, 2):
        raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: unsupported Predictor {predictor}")
    bits = np.atleast_1d(_tag(page, "BitsPerSample", 0))
    itemsize = page.dtype.itemsize
    if not (len(bits) > 0 and itemsize in (1, 2) and all(b == 8 * itemsize for b in bits)
            and np.dtype(page.dtype).kind == "u"):
        raise ValueError(f"{getattr(page.parent, 'filename', 'image')}: fallback only supports uint8/uint16 samples")

    h = page.imagelength
    w = page.imagewidth
    spp = page.samplesperpixel
    rows_per_strip = page.rowsperstrip if page.rowsperstrip is not None else h
    rows_per_strip = max(1, min(rows_per_strip, h))
    offsets = page.dataoffsets
    bytecounts = page.databytecounts
    bo = page.parent.byteorder
    dt = np.dtype("u1" if itemsize == 1 else f"{bo}u2")
    planarconfig = getattr(page, "planarconfig", 1) or 1
    fh = page.parent.filehandle

    def _decode_strip(strip_idx: int, rh: int, channels: int) -> np.ndarray:
        expected_bytes = rh * w * channels * itemsize
        fh.seek(offsets[strip_idx])
        raw = fh.read(bytecounts[strip_idx])
        if len(raw) < bytecounts[strip_idx]:
            raise ValueError(f"truncated strip {strip_idx} in TIFF file")
        if compression == 5:
            dec = _decode_tiff_lzw(raw, expected_bytes)
        elif compression == 32773:
            dec = _decode_packbits(raw, expected_bytes)
        else:
            try:
                dec = zlib.decompress(raw)
            except zlib.error as exc:
                raise ValueError(f"corrupt Deflate strip {strip_idx}: {exc}") from exc
            if len(dec) != expected_bytes:
                raise ValueError(f"Deflate decoded length {len(dec)} != expected {expected_bytes}")
        sarr = np.frombuffer(dec, dtype=dt).astype(page.dtype, copy=False)
        if channels > 1:
            sarr = sarr.reshape(rh, w, channels)
        else:
            sarr = sarr.reshape(rh, w)
        if predictor == 2:
            sarr = np.cumsum(sarr, axis=1, dtype=np.uint8 if itemsize == 1 else np.uint16)
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
                if (len(bits) > 0 and (all(b == 16 for b in bits) or all(b == 8 for b in bits))
                        and getattr(p, "dtype", None) is not None
                        and p.dtype.itemsize in (1, 2) and np.dtype(p.dtype).kind == "u"):
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


def _level(value, n, what, dtype=None, count=None):
    if value is None:
        return None
    v = np.atleast_1d(np.asarray(value, np.float64)).ravel()
    if dtype in (5, 10) and count is not None and v.size == 2 * int(count) and np.all(v[1::2] != 0):
        v = v[0::2] / v[1::2]
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


def _parse_tiff_eotf(icc_bytes, bits: int) -> tuple[str, float | None, np.ndarray | None]:
    """Read a single ICC TRC under the TIFF input policy; this is not an ICC colour transform."""
    try:
        b = bytes(icc_bytes) if icc_bytes is not None else b""
    except (TypeError, ValueError):
        b = b""
    if len(b) >= 132 and (bits == 8 or b[12:16] in (b"mntr", b"spac")):
        from .colorimetry import _trc as _icc_trc

        count = struct.unpack_from(">I", b, 128)[0]
        for i in range(min(count, 4096)):
            off = 132 + 12 * i
            if off + 12 > len(b):
                break
            sig, start, size = struct.unpack_from(">4sII", b, off)
            if sig not in (b"gTRC", b"rTRC", b"kTRC"):
                continue
            if size < 12 or start + size > len(b):
                continue
            tag = b[start:start + size]
            if tag[:4] == b"curv":
                n = struct.unpack_from(">I", tag, 8)[0]
                if n == 0:
                    return "linear", 1.0, None
                if n == 1 and len(tag) >= 14:
                    gamma = struct.unpack_from(">H", tag, 12)[0] / 256.0
                    return ("gamma", gamma, None) if gamma > 0 else ("linear", 1.0, None)
                if n >= 2 and 12 + 2 * n <= len(tag):
                    curve = np.frombuffer(tag, dtype=">u2", count=n, offset=12).astype(np.float32) / 65535.0
                    if np.all(np.diff(curve) >= 0):
                        return "lut", None, curve
            elif tag[:4] == b"para" and len(tag) >= 16:
                function = struct.unpack_from(">H", tag, 8)[0]
                gamma = struct.unpack_from(">i", tag, 12)[0] / 65536.0
                if function == 0:
                    if gamma > 0:
                        return "gamma", gamma, None
                    continue
                if function == 3 and len(tag) >= 32:
                    g, a, b_p, c, d = (struct.unpack_from(">i", tag, 12 + 4 * j)[0] / 65536.0 for j in range(5))
                    if (abs(g - 2.4) < 0.02 and abs(a - 1.0 / 1.055) < 0.01
                            and abs(b_p - 0.055 / 1.055) < 0.01 and abs(c - 1.0 / 12.92) < 0.01
                            and abs(d - 0.04045) < 0.01):
                        return "srgb", 2.4, None
                try:
                    fn = _icc_trc(tag)
                    curve = np.asarray(fn(np.linspace(0.0, 1.0, 4096, dtype=np.float64)), dtype=np.float32)
                    if np.all(np.isfinite(curve)) and np.all(np.diff(curve) >= 0):
                        return "lut", None, curve
                except ValueError:
                    continue
    return ("srgb", 2.4, None) if bits == 8 else ("linear", 1.0, None)


def _reconstruct_prestretched_rgb(rgb: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """Estimate a shared pedestal and clipped R tail using the task's cross-channel/gamma assumptions."""
    sub = rgb[::4, ::4] if rgb.shape[0] >= 32 and rgb.shape[1] >= 32 else rgb
    r_s, g_s, b_s = sub[..., 0], sub[..., 1], sub[..., 2]
    b_p01, r_p01 = float(np.percentile(b_s, 1.0)), float(np.percentile(r_s, 1.0))
    if not (b_p01 < 0.03 and r_p01 > 0.40):
        return rgb, None
    g_p95, b_p95 = float(np.percentile(g_s, 95.0)), float(np.percentile(b_s, 95.0))
    g_p01 = float(np.percentile(g_s, 1.0))
    unclip = (r_s < 0.998) & (g_p95 - g_s > 0.03)
    if int(unclip.sum()) >= 100:
        slope_rg = float(np.clip(np.median((1.0 - r_s[unclip]) / (g_p95 - g_s[unclip])), 0.35, 1.20))
        boundary = (r_s >= 0.985) & (r_s < 0.999)
        g_bound = float(np.median(g_s[boundary])) if int(boundary.sum()) >= 20 else g_p95
        delta_r = float(np.clip(slope_rg * max(0.0, g_p95 - g_bound), 0.0, 0.25))
    else:
        slope_rg, delta_r = 0.65, 0.08
    r_base_est = 1.0 + delta_r
    best_p0, best_err = 0.0, 1e9
    for candidate in np.linspace(0.02, 0.55, 107):
        dg = np.log10(max(g_p95 + candidate, 1e-4) / max(g_p01 + candidate, 1e-4))
        db = np.log10(max(b_p95 + candidate, 1e-4) / max(b_p01 + candidate, 1e-4))
        error = abs(db / max(dg, 1e-4) - 1.18)
        if error < best_err:
            best_err, best_p0 = error, float(candidate)
    scale, p0 = np.float32(r_base_est + best_p0), np.float32(best_p0)
    out = np.empty_like(rgb, dtype=np.float32)
    for r0 in range(0, rgb.shape[0], 256):
        sl = rgb[r0:r0 + 256]
        r_pred = np.clip(np.float32(r_base_est) - np.float32(slope_rg)
                         * np.maximum(np.float32(0.0), np.float32(g_p95) - sl[..., 1]),
                         np.float32(1.0), np.float32(r_base_est))
        weight = np.clip((sl[..., 0] - np.float32(0.990)) / np.float32(0.009), 0.0, 1.0)
        r_rec = (np.float32(1.0) - weight) * sl[..., 0] + weight * r_pred
        out[r0:r0 + 256, :, 0] = np.power(np.maximum((r_rec + p0) / scale, np.float32(0.0)), np.float32(2.2))
        out[r0:r0 + 256, :, 1] = np.power(np.maximum((sl[..., 1] + p0) / scale, np.float32(0.0)), np.float32(2.2))
        out[r0:r0 + 256, :, 2] = np.power(np.maximum((sl[..., 2] + p0) / scale, np.float32(0.0)), np.float32(2.2))
    return out, dict(r_base_est=round(r_base_est, 4), slope_rg=round(slope_rg, 4),
                     pedestal_p0=round(best_p0, 4), gamma_eotf=2.2)


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
            raise ValueError(f"{path.name}: no 1-, 3- or 4-sample LinearRaw/8-/16-bit scanner image")
        lut_tag = main.tags.get("LinearizationTable")
        lut = None
        if lut_tag is not None:
            lut = np.atleast_1d(np.asarray(lut_tag.value, np.float64)).ravel()
            if lut.size < 2 or not np.all(np.isfinite(lut)) or np.any(np.diff(lut) < 0):
                raise ValueError(f"{path.name}: LinearizationTable must be finite and non-decreasing")
        bits = _tag(main, "BitsPerSample")
        bits = int(np.atleast_1d(bits)[0])
        n_samples = main.samplesperpixel
        bl_tag, wl_tag = main.tags.get("BlackLevel"), main.tags.get("WhiteLevel")
        black = _level(bl_tag.value if bl_tag else None, n_samples, "BlackLevel",
                       getattr(bl_tag, "dtype", None), getattr(bl_tag, "count", None))
        white = _level(wl_tag.value if wl_tag else None, n_samples, "WhiteLevel",
                       getattr(wl_tag, "dtype", None), getattr(wl_tag, "count", None))
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
        mode, gamma, curve = "linear", 1.0, None
        if allow_tiff and lut is None:
            icc = _tag(main, "InterColorProfile")
            if icc is None:
                icc = _tag(ifd0, "InterColorProfile")
            mode, gamma, curve = _parse_tiff_eotf(icc, bits)
    if a.ndim == 2:
        rgb_raw = np.repeat(a[:, :, None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        rgb_raw = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] in (3, 4):
        rgb_raw = a[:, :, :3]
    else:
        raise ValueError(f"{path.name}: unexpected image shape {a.shape}")
    sat_ch = rgb_raw >= white
    r_sat, g_sat, b_sat = (float(sat_ch[:, :, i].mean()) for i in range(3))
    if (allow_tiff and n_samples >= 3
            and ((r_sat > 0.30 and g_sat < 0.05 and b_sat < 0.05)
                 or (r_sat > 0.15 and g_sat < 0.002 and b_sat < 0.002))):
        # T4u: a pre-stretched orange mask can clip R while G and B retain usable signal
        sat = sat_ch[:, :, 1] | sat_ch[:, :, 2]
        meta["single_channel_prestretched"] = "R"
    else:
        sat = np.any(sat_ch, -1)
    del sat_ch
    rgb = rgb_raw.astype(np.float32, copy=True)
    del a, rgb_raw
    if black != 0.0:
        rgb -= np.float32(black)
    rgb /= np.float32(white - black)
    if allow_tiff and lut is None:
        reconstruction = None
        if bits == 16 and meta.get("single_channel_prestretched") == "R" and mode == "linear":
            rgb, reconstruction = _reconstruct_prestretched_rgb(rgb)
        if reconstruction is not None:
            meta["prestretched_reconstruction"] = reconstruction
        elif mode != "linear":
            for r0 in range(0, rgb.shape[0], 256):
                sl = np.clip(rgb[r0:r0 + 256], 0.0, 1.0)
                if mode == "srgb":
                    rgb[r0:r0 + 256] = _srgb_eotf(sl)
                elif mode == "gamma" and abs(gamma - 1.0) > 0.05:
                    rgb[r0:r0 + 256] = np.power(sl, np.float32(gamma))
                elif mode == "lut":
                    rgb[r0:r0 + 256] = np.interp(sl, np.linspace(0.0, 1.0, len(curve)), curve)
            if mode == "srgb":
                meta["tiff_eotf"] = "srgb"
            elif mode == "gamma" and abs(gamma - 1.0) > 0.05:
                meta["tiff_eotf"] = f"gamma:{gamma:.4f}"
            elif mode == "lut":
                meta["tiff_eotf"] = f"icc_lut:{len(curve)}"
    if not _all_finite(rgb):
        raise ValueError(f"{path.name}: non-finite samples")
    return ScannerFrame(rgb=rgb, sat=sat, black=black, white=white, meta=meta)


def block_means(rgb, sat, block: int):
    """(hb, wb, 3) block means and (hb, wb) any-saturated flags; partial edge blocks dropped."""
    from .invert import CHUNK_ROWS, map_row_blocks

    hb, wb = rgb.shape[0] // block, rgb.shape[1] // block
    if hb == 0 or wb == 0:
        raise ValueError(f"image {rgb.shape[:2]} smaller than one {block}px block")
    h, w = hb * block, wb * block
    inv_area = np.float64(1.0 / (block * block))
    means = np.empty((hb, wb, 3), dtype=np.float32)
    any_sat = np.empty((hb, wb), dtype=bool)
    step = max(block, (CHUNK_ROWS // block) * block)

    def _run(r0: int) -> None:
        r1 = min(h, r0 + step)
        b0, b1 = r0 // block, r1 // block
        blk = rgb[r0:r1, :w].reshape(b1 - b0, block, wb, block, 3)
        means[b0:b1] = (blk.sum(axis=3, dtype=np.float64).sum(axis=1) * inv_area).astype(np.float32)
        any_sat[b0:b1] = sat[r0:r1, :w].reshape(b1 - b0, block, wb, block).any(axis=3).any(axis=1)

    map_row_blocks(h, _run, step)
    return means, any_sat


def estimate_base_scanner(frame: ScannerFrame, *, block: int = 24, **kwargs):
    """Automatic film base on the scanner image (same block selection as the camera path)."""
    sp, sp_sat = block_means(frame.rgb, frame.sat, block)
    kwargs.setdefault("exclude_near_saturated", False)
    return estimate_base_blocks(sp, sp_sat, block=block, **kwargs)
