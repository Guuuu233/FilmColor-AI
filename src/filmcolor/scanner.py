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
from .frozen import check_name

LINEAR_RAW = 34892
SILVERFAST_KEYS = ("General_NegativeMode", "isWithIR", "MultiExposure", "General_HdrMode",
                   "IsLayerActive_ISRD", "IsLayerActive_SRDX", "ImageFormat", "General_Bitdepth")


@dataclass
class ScannerFrame:
    rgb: np.ndarray        # (H, W, 3) float32, black-subtracted, / (white - black)
    sat: np.ndarray        # (H, W) bool, any channel at or above white
    black: float
    white: float
    meta: dict


def _all_pages(tf):
    for p in tf.pages:
        yield p
        for sp in (p.pages or []) if hasattr(p, "pages") else []:
            yield sp


def _tag(page, name, default=None):
    t = page.tags.get(name)
    return default if t is None else t.value


def _main_page(tf):
    for n_samples in ((3, 4), (1,)):
        cands = [p for p in _all_pages(tf) if p.photometric == LINEAR_RAW and p.samplesperpixel in n_samples
                 and not (_tag(p, "NewSubfileType", 0) or 0) & 1]
        if cands:
            return max(cands, key=lambda p: p.shape[0] * p.shape[1])
    return None


def is_scanner_dng(path) -> bool:
    """True for a DNG whose main image is 1-, 3- or 4-sample LinearRaw (no CFA)."""
    if Path(path).suffix.lower() != ".dng":
        return False
    with tifffile.TiffFile(path) as tf:
        return _main_page(tf) is not None


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
    check_name(path, allow_frozen=allow_frozen)  # sealed dataset frames, also before any hash exists
    if not path.is_file():
        raise FileNotFoundError(path)
    with tifffile.TiffFile(path) as tf:
        main = _main_page(tf)
        if main is None:
            raise ValueError(f"{path.name}: no 1-, 3- or 4-sample LinearRaw image (not a linear scanner DNG)")
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
        a = main.asarray()
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
            m = ir[0].asarray()[::8, ::8].astype(np.float64)
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
    sat = np.any(rgb_raw >= white, -1)
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
