"""High-speed, low-RAM local HTTP bridge for FilmColor Lightroom Classic (.lrplugin) and Photoshop (UXP) plugins.

Performance architecture:
- Preview open: single rawpy open + CFA base (+ inline sprocket-backlight guard) + half_size linear decode
  (exact 2x2 Bayer cell mean) -> block white point -> T24 auto estimate (~1.5-3.5 s vs 18-58 s).
- Full-res convert: reuses cached preview estimation when available (0 s overhead), opens RAW once,
  crops before inversion/rendering when Lightroom crop is active, runs chunked multi-threaded float32
  fused inversion + T24 render + 33^3 float32 trilinear darkroom LUT + fast zlib-1/predictor TIFF.
- Clean lifecycle: POST /shutdown (on Lightroom exit) + automatic 30-min idle shutdown.
"""
from __future__ import annotations

import argparse
import atexit
import base64
from contextlib import contextmanager
import gc
import io as _io
import json
import math
import os
from pathlib import Path
from fractions import Fraction
import secrets
import select
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import rawpy

from . import __version__
from .base import (
    base_deviation,
    block48_L_p01,
    decoupled_quantile_base,
    estimate_base_blocks,
    estimate_base_roll,
    natural_roll_base,
)
from .compare import block_mean
from .darkroom import (
    DarkroomGrade,
    Geometry,
    T24_STYLE_PRESETS,
    analyze_scene_metrics,
    resolve_adaptive_preset,
    _get_pool,
    apply_darkroom_grade,
    apply_geometry,
    apply_lr_crop_angle,
    apply_vignette,
    export_image,
    free_stem_ext,
)
from .holder import film_box, holder_mask
from .invert import (
    CHUNK_ROWS,
    InversionParams,
    block_white_point,
    blue_white_guard, self_ratio_white_point,
    _t18_affine_params,
    _t18_chunk_from_s,
    central_region,
    clamp_spikes,
    region_mask,
)
from .noref import (
    NOREF_T23,
    NOREF_T24,
    NOREF_T25,
    NOREF_V3,
    STYLE_PRESETS,
    estimation_input,
    pool_roll_params_t24,
    preset_config,
)
from .pipeline import (
    FILM_MODES,
    T19_FAMILY,
    coerce_render_params,
    neutralise_bw_params,
    run_auto_estimate,
    sha256,
)
from .preview import Preview, input_warnings, load_preview
from .rawpy_io import _open as open_rawpy, _postprocess_linear
from .render import (
    BOUNDS,
    EXTENDED_BOUNDS,
    RenderParams,
    RenderParamsT19,
    RenderParamsT24,
    T19_BOUNDS,
    T24_BOUNDS,
    _prepare_t19_lut,
    _render_t19_core,
    _render_t24_core,
    render_fast_full,
    render_t19,
    render_t24,
    _t24_linear_matrix,
    camera_color_matrix,
    render as render_image,
    srgb_eotf,
)
from .scanner import block_means, is_scanner_dng, load_scanner

ENGINES = ("t25", "t24", "t23", "v3")
T25_ENGINES = ("t25", "t24")
T24_CAM_MATRIX_MIX = 0.35
WHITE_BLOCK, WHITE_PCT = 16, 99.8
SPIKE_CLAMP = (1.15, 0.12)
WHITE_SELF_RATIO = None
WHITE_B_GUARD = (1.06, 1.18, 0.65, 0.50)
EXIFTOOL_CANDIDATES = ("/opt/homebrew/bin/exiftool", "/usr/local/bin/exiftool")


def _cache_root() -> Path:
    env = os.environ.get("FILMCOLOR_DATA_ROOT")
    d = (Path(env) / "app_cache") if (env and Path(env).is_dir()) else (Path(tempfile.gettempdir()) / "filmcolor_app_cache")
    d.mkdir(parents=True, exist_ok=True)
    return d


def png_bytes(img: np.ndarray, *, compress_level: int = 1) -> bytes:
    """8-bit RGB PNG (no filter, zero-copy buffer; compress_level=0 for ultra-fast local temp previews)."""
    h, w = img.shape[:2]
    raw = np.empty((h, w * 3 + 1), np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = np.ascontiguousarray(img[..., :3], np.uint8).reshape(h, -1)
    chunk = lambda kind, data: (struct.pack(">I", len(data)) + kind + data
                                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(memoryview(raw), int(compress_level))) + chunk(b"IEND", b""))


def jpeg_bytes(img: np.ndarray, quality: int = 92) -> bytes:
    """Fast 8-bit RGB JPEG for interactive preview streaming."""
    from PIL import Image

    buf = _io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img[..., :3], np.uint8), mode="RGB").save(
        buf, format="JPEG", quality=quality, subsampling=0 if quality >= 95 else 2
    )
    return buf.getvalue()


def to8(display: np.ndarray) -> np.ndarray:
    out = np.clip(np.asarray(display, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    out *= np.float32(255.0)
    out += np.float32(0.5)
    return out.astype(np.uint8)


def _sanitize_json_obj(obj):
    """Recursively replace non-finite floats (NaN/Inf) with None for strict JSON compliance."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_obj(v) for v in obj]
    return obj


def _parse_max_dim(val) -> int | None:
    if val is None or val == "":
        return None
    fv = float(val)
    if not math.isfinite(fv) or fv <= 0:
        return None
    return int(np.clip(int(fv), 16, 4096))


def _scanner_shape(path: Path) -> tuple[int, int]:
    """Read scanner TIFF/DNG (H, W) from file header without full pixel decode."""
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            for page in tf.pages:
                sh = page.shape
                if len(sh) >= 2 and sh[0] >= 64 and sh[1] >= 64:
                    return int(sh[0]), int(sh[1])
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.height), int(im.width)
    except Exception:  # noqa: BLE001
        pass
    fr = load_scanner(path, allow_frozen=True)
    sh = (int(fr.rgb.shape[0]), int(fr.rgb.shape[1]))
    del fr
    return sh


def _preview_has_border(pv: Preview) -> bool:
    cached = getattr(pv, "_cached_has_border", None)
    if cached is None:
        cached = bool(pv.holder is not None and film_box(pv.holder) is not None)
        pv._cached_has_border = cached
    return cached


def negative_view(path: Path, max_side: int = 1600, expected_shape: tuple[int, int] | None = None) -> tuple[bytes, str]:
    """The negative as captured: camera embedded JPEG preview (or half-size fallback), or downscaled scanner DNG."""
    if not is_scanner_dng(path):
        with open_rawpy(path) as r:
            try:
                t = r.extract_thumb()
                if t.format == rawpy.ThumbFormat.JPEG:
                    raw_data = bytes(t.data)
                    if expected_shape is not None and expected_shape[0] != expected_shape[1]:
                        from PIL import Image
                        with Image.open(_io.BytesIO(raw_data)) as tim:
                            tw, th = tim.size
                        exp_land = expected_shape[1] > expected_shape[0]
                        if (tw > th) != exp_land:
                            raise ValueError("embedded thumbnail orientation disagrees with sensor-native shape")
                    return raw_data, "image/jpeg"
                arr_t = np.asarray(t.data)
                if expected_shape is not None and expected_shape[0] != expected_shape[1]:
                    th, tw = arr_t.shape[:2]
                    if (tw > th) != (expected_shape[1] > expected_shape[0]):
                        raise ValueError("embedded bitmap orientation disagrees with sensor-native shape")
                return png_bytes(arr_t), "image/png"
            except Exception:  # noqa: BLE001 — fallback when RAW has no matching embedded thumbnail
                rgb = r.postprocess(
                    half_size=True, no_auto_bright=True, output_bps=8,
                    use_camera_wb=False, use_auto_wb=False, user_wb=[1, 1, 1, 1],
                    user_flip=0,
                )
                return png_bytes(rgb), "image/png"
    rgb = load_scanner(path).rgb
    f = max(1, -(-max(rgb.shape[:2]) // max_side))
    small = block_mean(rgb, f) if f > 1 else rgb
    return png_bytes(to8((small / max(np.percentile(small, 99.9), 1e-6)) ** (1 / 2.2))), "image/png"


def _exiftool() -> str | None:
    found = shutil.which("exiftool")
    return found or next((c for c in EXIFTOOL_CANDIDATES if os.access(c, os.X_OK)), None)


def _exif_info(paths: list[Path]) -> dict[Path, dict]:
    """EXIF ExposureTime (1.0 when absent, e.g. scanner DNGs), camera Model and capture date per file."""
    out: dict[Path, dict] = {}
    need_tool: list[Path] = []
    try:
        import exifread
    except ImportError:
        exifread = None
    for p in paths:
        info = dict(exposure=None, model=None, date=None)
        if exifread is not None:
            try:
                with open(p, "rb") as fh:
                    tags = exifread.process_file(fh, details=False, stop_tag="DateTimeOriginal")
                r = tags.get("EXIF ExposureTime")
                if r and r.values:
                    v = float(Fraction(r.values[0].num, r.values[0].den))
                    if np.isfinite(v) and v > 0:
                        info["exposure"] = v
                model, date = tags.get("Image Model"), tags.get("EXIF DateTimeOriginal")
                info["model"] = str(model).strip() or None if model else None
                info["date"] = str(date)[:10] or None if date else None
            except Exception:  # noqa: BLE001
                pass
        out[p] = info
        if info["exposure"] is None:
            need_tool.append(p)
    tool = _exiftool() if need_tool else None
    if tool:
        try:
            kw_win = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if sys.platform == "win32" else {}
            proc = subprocess.run(
                [tool, "-j", "-n", "-q", "-q", "-ExposureTime", "-Model", "-DateTimeOriginal", "-CreateDate",
                 *(str(p) for p in need_tool)], capture_output=True, text=True, check=False, **kw_win,
            )
            by_src = {Path(d["SourceFile"]): d for d in json.loads(proc.stdout or "[]")}
        except Exception:  # noqa: BLE001
            by_src = {}
        for p in need_tool:
            d = by_src.get(p, {})
            v = d.get("ExposureTime")
            if isinstance(v, (int, float)) and np.isfinite(v) and v > 0:
                out[p]["exposure"] = float(v)
            if d.get("Model") is not None:
                out[p]["model"] = out[p]["model"] or str(d["Model"]).strip() or None
            date = d.get("DateTimeOriginal") or d.get("CreateDate")
            if date is not None:
                out[p]["date"] = out[p]["date"] or str(date)[:10] or None
    for info in out.values():
        if info["exposure"] is None:
            info["exposure"] = 1.0
    return out


def _scene_metrics_for_preview(pv: Preview) -> dict[str, float] | None:
    """Compute & cache lightweight scene metrics once per Preview from its locked auto render (~4 ms)."""
    cached = getattr(pv, "_scene_metrics", None)
    if cached is not None:
        return cached
    try:
        raw_p = (pv.auto or {}).get("params")
        if not raw_p:
            return None
        rp = coerce_render_params(dict(raw_p))
        disp, stats = pv.render_stats(rp)
        # When region_source is central_default (only central 50%), measure scene metrics across
        # the full 90% interior frame so outer foregrounds (e.g. grass below sky) are not missed.
        if getattr(pv, "region_source", "") == "central_default":
            h, w = disp.shape[:2]
            my, mx = max(1, int(round(h * 0.05))), max(1, int(round(w * 0.05)))
            sm_mask = np.zeros((h, w), dtype=bool)
            sm_mask[my:h - my, mx:w - mx] = True
        else:
            sm_mask = pv.mask
        metrics = analyze_scene_metrics(disp, sm_mask, above_white=stats.get("above_white"))
        pv._scene_metrics = metrics
        return metrics
    except Exception:  # noqa: BLE001
        return None


def _apply_t24_preset(
    base_params: dict, preset: str, film: str, engine: str, *, raw_matrix=None, scene_metrics: dict | None = None
) -> tuple[dict, dict]:
    """Apply an adaptive darkroom style preset cleanly on top of locked auto params."""
    spec = resolve_adaptive_preset(preset, film=film, scene_metrics=scene_metrics)
    bounds = T24_BOUNDS if engine in T25_ENGINES else T19_BOUNDS
    p = dict(base_params)
    mix = spec.get("camera_matrix_mix")
    if mix is not None and film == "color" and raw_matrix is not None:
        try:
            eff_mix = float(mix)
            m = camera_color_matrix(raw_matrix, eff_mix)
            # Sensor-normalized boost cap: steep camera matrices (e.g. Panasonic/Sony)
            # would otherwise push chromatic gain ~2x harder than Nikon and shift T24 v4 WB.
            max_diag = max(float(m[0][0]), float(m[1][1]), float(m[2][2]))
            if max_diag > 1.28 and eff_mix > T24_CAM_MATRIX_MIX:
                m_base = camera_color_matrix(raw_matrix, T24_CAM_MATRIX_MIX)
                base_diag = max(float(m_base[0][0]), float(m_base[1][1]), float(m_base[2][2]))
                if max_diag > base_diag + 1e-6:
                    t_cap = max(0.0, min(1.0, (1.28 - base_diag) / (max_diag - base_diag)))
                    eff_mix = T24_CAM_MATRIX_MIX + t_cap * (eff_mix - T24_CAM_MATRIX_MIX)
                    m = camera_color_matrix(raw_matrix, eff_mix)
            p["matrix"] = [list(r) for r in m]
        except ValueError:
            pass
    color_keys = {"temp", "tint", "sat", "slope_r", "slope_b"}
    for k, dv in spec["offsets"].items():
        if k not in bounds:
            continue
        if film == "bw" and k in color_keys:
            continue
        lo, hi = bounds[k]
        p[k] = float(np.clip(float(p.get(k, 0.0)) + float(dv), lo, hi))
    grade = dict(spec["grade"])
    if film == "bw":
        for ck in ("vibrance", "skin_protect", "hsl"):
            grade.pop(ck, None)
    return p, grade


DEFAULT_PORT = 18765
PREVIEW_MAX_SIDE = 960
DEFAULT_IDLE_TIMEOUT = 1800

LR_ORIENT_TO_ROT90 = {
    "AB": 0,  # EXIF 1: 0 deg normal
    "BA": 0,  # EXIF 2: mirror horizontal
    "CD": 2,  # EXIF 3: 180 deg
    "DC": 2,  # EXIF 4: mirror vertical (= rot180 + mirror horizontal)
    "AD": 3,  # EXIF 5: transpose (= rot90 CW + mirror horizontal)
    "BC": 3,  # EXIF 6: 90 deg CW
    "CB": 1,  # EXIF 7: transverse (= rot90 CCW + mirror horizontal)
    "DA": 1,  # EXIF 8: 90 deg CCW
}
LR_ORIENT_FLIP_H = {"BA", "DC", "AD", "CB"}


def _postprocess_half_u16(r) -> np.ndarray:
    """Fast half-size (2x2 Bayer cell) linear uint16 camera RGB directly from LibRaw."""
    import rawpy
    return r.postprocess(
        half_size=True,
        output_color=rawpy.ColorSpace.raw,
        gamma=(1, 1),
        no_auto_bright=True,
        output_bps=16,
        use_camera_wb=False,
        use_auto_wb=False,
        user_wb=[1.0, 1.0, 1.0, 1.0],
        user_flip=0,
        adjust_maximum_thr=0.0,
        highlight_mode=rawpy.HighlightMode.Clip,
    )


def _postprocess_half_linear(r) -> np.ndarray:
    """Fast half-size (2x2 Bayer cell) linear camera RGB matching _postprocess_linear scale exactly."""
    out = _postprocess_half_u16(r).astype(np.float32)
    out /= np.float32(65535.0)
    return out


def _resize_linear_f32(img: np.ndarray, nw: int, nh: int) -> np.ndarray:
    """Separable area+bilinear float32 resize preserving HDR (<0 / >1) linear values without clipping."""
    h, w = img.shape[:2]
    if h == nh and w == nw:
        return img
    arr = img
    # Pre-decimate by 2x2 box mean while >= 2x target to prevent aliasing and speed up bilinear
    while arr.shape[0] >= 2 * nh and arr.shape[1] >= 2 * nw:
        arr = _block_mean2_f32(arr)
    ch, cw = arr.shape[:2]
    if ch == nh and cw == nw:
        return arr
    # Separable bilinear: vertical then horizontal
    ys = np.linspace(0.0, max(ch - 1, 0), nh, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.minimum(y0 + 1, ch - 1)
    wy = (ys - y0)[:, None, None]
    tmp = arr[y0] * (np.float32(1.0) - wy) + arr[y1] * wy
    xs = np.linspace(0.0, max(cw - 1, 0), nw, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.minimum(x0 + 1, cw - 1)
    wx = (xs - x0)[None, :, None]
    return np.ascontiguousarray(tmp[:, x0] * (np.float32(1.0) - wx) + tmp[:, x1] * wx, dtype=np.float32)


def _resize_mask_nn(mask: np.ndarray, nw: int, nh: int) -> np.ndarray:
    h, w = mask.shape[:2]
    if h == nh and w == nw:
        return mask
    ys = np.linspace(0, max(h - 1, 0), nh).round().astype(np.int32)
    xs = np.linspace(0, max(w - 1, 0), nw).round().astype(np.int32)
    return np.ascontiguousarray(mask[ys][:, xs])


def _block_mean2_f32(img: np.ndarray) -> np.ndarray:
    """Fast 2x2 float32 block mean (~6x faster than generic reshape().mean(axis=(1,3)))."""
    h, w = (img.shape[0] // 2) * 2, (img.shape[1] // 2) * 2
    out = img[0:h:2, 0:w:2] + img[0:h:2, 1:w:2]
    out += img[1:h:2, 0:w:2]
    out += img[1:h:2, 1:w:2]
    out *= np.float32(0.25)
    return out


def _read_raw_xyz_matrix(path: Path):
    if is_scanner_dng(path, allow_frozen=True):
        return None
    try:
        import rawpy
        with rawpy.imread(str(path)) as r:
            return np.asarray(r.rgb_xyz_matrix)[:3].tolist()
    except Exception:  # noqa: BLE001
        return None


def _read_camera_matrix(path: Path, film: str, mix: float, *, raw_m=None):
    if film != "color":
        return None
    if raw_m is None:
        raw_m = _read_raw_xyz_matrix(path)
    if raw_m is None:
        return None
    try:
        return camera_color_matrix(raw_m, mix)
    except Exception:  # noqa: BLE001
        return None


def _normalize_lr_crop(lr_crop) -> tuple[float, float, float, float] | None:
    if not lr_crop or len(lr_crop) != 4:
        return None
    try:
        t, l, b, r = (float(x) for x in lr_crop)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(x) for x in (t, l, b, r)):
        return None
    t = max(0.0, min(0.95, t))
    l = max(0.0, min(0.95, l))
    b = max(t + 0.05, min(1.0, b))
    r = max(l + 0.05, min(1.0, r))
    if t <= 1e-3 and l <= 1e-3 and b >= 1.0 - 1e-3 and r >= 1.0 - 1e-3:
        return None
    return (t, l, b, r)


def _lr_crop_to_image_region(lr_crop: tuple[float, float, float, float], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    H, W = int(shape[0]), int(shape[1])
    t, l, b, r = lr_crop
    y0 = max(0, min(H - 16, int(round(t * H))))
    x0 = max(0, min(W - 16, int(round(l * W))))
    y1 = max(y0 + 16, min(H, int(round(b * H))))
    x1 = max(x0 + 16, min(W, int(round(r * W))))
    return (y0, x0, y1 - y0, x1 - x0)


def _auto_cfg(path: Path, film: str, engine: str, preset: str, *, raw_m=None) -> dict:
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    if engine in ("t25", "t24", "t23"):
        p_name = preset if preset in T24_STYLE_PRESETS else "standard"
        base_cfg = NOREF_T25 if engine == "t25" else (NOREF_T24 if engine == "t24" else NOREF_T23)
        cfg = dict(base_cfg, preset=p_name, camera_matrix_mix=T24_CAM_MATRIX_MIX)
        m = _read_camera_matrix(path, film, T24_CAM_MATRIX_MIX, raw_m=raw_m)
        return dict(cfg, matrix=m) if m is not None else cfg
    p_name = preset if preset in STYLE_PRESETS else "standard"
    cfg = preset_config(p_name)
    m = _read_camera_matrix(path, film, cfg["camera_matrix_mix"], raw_m=raw_m)
    return dict(cfg, matrix=m) if m is not None else cfg


def _apply_user_adjustments(
    auto_params: dict,
    auto_grade: dict,
    adj: dict | None,
    *,
    engine: str,
    film: str,
) -> tuple[dict, DarkroomGrade]:
    adj = adj or {}
    p = dict(auto_params)
    g = dict(auto_grade)
    is_t19 = engine in ("t25", "t24", "t23")
    bounds = T24_BOUNDS if engine in T25_ENGINES else (T19_BOUNDS if engine == "t23" else (BOUNDS | EXTENDED_BOUNDS))

    delta_map = {
        "ev": 1.0, "temp": 0.01, "tint": 0.01, "contrast": 0.008,
        "sat": 0.01, "shoulder": 0.005, "toe": 0.01,
    }
    color_only = {"temp", "tint", "sat"}
    for key, scale in delta_map.items():
        dv = float(adj.get(key, 0.0) or 0.0)
        if not dv or key not in bounds:
            continue
        if film == "bw" and key in color_only:
            continue
        lo, hi = bounds[key]
        p[key] = float(np.clip(float(p.get(key, 0.0)) + dv * scale, lo, hi))

    grade_keys = (
        "highlights", "shadows", "whites", "blacks", "vibrance",
        "skin_protect", "clarity", "dehaze", "sharpen", "vignette", "grain",
    )
    for k in grade_keys:
        v = float(adj.get(k, 0.0) or 0.0)
        if not v:
            continue
        if film == "bw" and k in ("vibrance", "skin_protect"):
            continue
        val = v / 100.0
        if not is_t19 and k in ("highlights", "shadows", "whites", "blacks", "vibrance", "skin_protect"):
            lo, hi = EXTENDED_BOUNDS[k]
            p[k] = float(np.clip(float(p.get(k, 0.0)) + val, lo, hi))
        else:
            lo = 0.0 if k in ("skin_protect", "sharpen", "grain") else -1.0
            g[k] = float(np.clip(float(g.get(k, 0.0)) + val, lo, 1.0))

    if isinstance(adj.get("grade"), dict):
        for k, v in adj["grade"].items():
            g[k] = v

    if film == "bw":
        p, _ = neutralise_bw_params(p)
    return p, DarkroomGrade.from_dict(g)


def _auto_border_crop_norm(pv: Preview, pad: float = 0.008) -> tuple[float, float, float, float] | None:
    if pv.holder is None or not pv.holder.any():
        return None
    fb = film_box(pv.holder)
    if fb is None:
        return None
    ft, fl, fh, fw = fb
    H, W = pv.positive.shape[:2]
    t = max(0.0, min(0.45, ft / H + pad))
    l = max(0.0, min(0.45, fl / W + pad))
    b = min(1.0, max(t + 0.1, (ft + fh) / H - pad))
    r = min(1.0, max(l + 0.1, (fl + fw) / W - pad))
    if t <= 0.01 and l <= 0.01 and b >= 0.99 and r >= 0.99:
        return None
    return (t, l, b, r)


def _sprocket_guard_from_sp(sp, sp_sat, crop, init_rgb, g_density_p01: float, film: str):
    """Inline sprocket-backlight guard reusing already-computed (sp, sp_sat) — zero extra RAW decode!"""
    if film != "color" or not init_rgb or len(init_rgb) != 3 or init_rgb[1] <= 1e-6:
        return None
    rg1, bg1 = float(init_rgb[0]) / float(init_rgb[1]), float(init_rgb[2]) / float(init_rgb[1])
    if not (rg1 < 0.78 and bg1 > 0.44) or g_density_p01 < 0.40:
        return None
    try:
        hb, wb, _ = sp.shape
        valid = ~sp_sat & np.all(sp > 0, -1)
        if crop is not None:
            t, l, h, w = crop
            rows, cols = np.arange(hb) * 24, np.arange(wb) * 24
            valid &= ((rows >= t) & (rows + 24 <= t + h))[:, None] & ((cols >= l) & (cols + 24 <= l + w))[None, :]
        if int(valid.sum()) < 100:
            return None
        Y = sp.mean(-1)
        p90 = float(np.quantile(Y[valid], 0.90))
        est_film = estimate_base_blocks(sp, sp_sat | (Y > 2.5 * p90), block=12 if crop is None else 24, crop=crop, exclude_near_saturated=True)
        rgb2 = [float(x) for x in est_film.rgb]
        if rgb2[1] <= 1e-6 or rgb2[2] <= 1e-6:
            return None
        rg2, bg2 = rgb2[0] / rgb2[1], rgb2[2] / rgb2[1]
        b_ratio = float(init_rgb[2]) / rgb2[2]
        if b_ratio >= 2.8 and rg2 >= 1.30 * rg1 and bg2 <= 0.80 * bg1:
            return rgb2, dict(
                method="auto_sprocket_guard",
                raw_backlight_rgb=[float(x) for x in init_rgb],
                effective_rgb=rgb2,
                b_ratio=round(b_ratio, 3),
                g_density_p01_before=round(g_density_p01, 3),
            )
    except Exception:  # noqa: BLE001
        return None
    return None


def _fast_load_camera_preview(
    path: Path,
    *,
    max_side: int,
    film: str,
    engine: str,
    preset: str,
    base_mode: str,
    extra: dict,
    lr_crop: tuple[float, float, float, float] | None,
    preloaded: tuple | None = None,
) -> tuple[Preview, list | None]:
    """Ultra-fast single-open camera RAW preview: uses C-level LibRaw half_size=True directly for base,
    white-point, T24 estimation, and preview (15-30x faster than full AHD + CFA copy)."""
    if preloaded is not None:
        if len(preloaded) == 7:
            rgb4, sat_4, grid_div, raw_m, full_shape, sp, sp_sat = preloaded
        else:
            rgb4, grid_div, raw_m, full_shape, sp, sp_sat = preloaded
            sat_4 = (rgb4[..., 0] >= np.float32(0.999)) | (rgb4[..., 1] >= np.float32(0.999)) | (rgb4[..., 2] >= np.float32(0.999))
        hh, hw = full_shape[0] // 2, full_shape[1] // 2
    else:
        with open_rawpy(path) as r:
            raw_m = np.asarray(r.rgb_xyz_matrix)[:3].tolist()
            s_sz = r.sizes
            full_shape = (int(s_sz.height), int(s_sz.width))
            img_u16 = _postprocess_half_u16(r)

        hh, hw = img_u16.shape[:2]
        h2, w2 = (hh // 2) * 2, (hw // 2) * 2
        # Exact uint16 threshold for rgb_half >= 0.999 (ceil(0.999 * 65535) == 65470)
        sat_half = (img_u16[..., 0] >= 65470) | (img_u16[..., 1] >= 65470) | (img_u16[..., 2] >= 65470)
        sat_4 = (
            sat_half[0:h2:2, 0:w2:2]
            | sat_half[0:h2:2, 1:w2:2]
            | sat_half[1:h2:2, 0:w2:2]
            | sat_half[1:h2:2, 1:w2:2]
        )
        del sat_half
        # Direct uint16 2x2 sum -> float32 rgb4 (avoids allocating 72MB rgb_half float32!)
        s_u32 = img_u16[0:h2:2, 0:w2:2].astype(np.uint32)
        s_u32 += img_u16[0:h2:2, 1:w2:2]
        s_u32 += img_u16[1:h2:2, 0:w2:2]
        s_u32 += img_u16[1:h2:2, 1:w2:2]
        del img_u16
        rgb4 = s_u32.astype(np.float32)
        del s_u32
        rgb4 *= np.float32(0.25 / 65535.0)
        # 6x6 blocks on rgb4 == exact 12x12 blocks on half_size == exact 24x24 sensor blocks!
        sp, sp_sat = block_means(rgb4, sat_4, 6)
        grid_div = 4

    base_source = extra.get("base_source")
    if extra.get("base_rgb") is not None:
        base_rgb = [float(x) for x in extra["base_rgb"]]
        base_info = dict(method=(base_source or {}).get("method", "given"), rgb_render=base_rgb)
        inside_img = None
    else:
        est = estimate_base_blocks(sp, sp_sat, block=12, crop=None, exclude_near_saturated=True)
        base_rgb = [float(x) for x in est.rgb]
        t0, l0, h0, w0 = central_region((hh, hw))
        centres = [(ry + rh / 2, rx + rw / 2) for ry, rx, rh, rw in est.regions]
        inside_img = float(np.mean([t0 <= y < t0 + h0 and l0 <= x < l0 + w0 for y, x in centres]))
        base_info = dict(method="auto", estimate=est.to_dict(), rgb_render=base_rgb)

    h4, w4 = rgb4.shape[:2]

    if lr_crop is not None:
        img_reg_full = _lr_crop_to_image_region(lr_crop, full_shape)
        img_reg_4 = _lr_crop_to_image_region(lr_crop, (h4, w4))
        reg_src = "lr_crop"
    else:
        img_reg_full = central_region(full_shape)
        img_reg_4 = central_region((h4, w4))
        reg_src = "central_default"

    def _build_mask4(reg4):
        rm = region_mask((h4, w4), reg4)
        m_ns = (~sat_4) & rm
        return m_ns if int(m_ns.sum()) >= 64 else rm

    mask4 = _build_mask4(img_reg_4)
    wb4 = max(1, WHITE_BLOCK // grid_div)

    def _fast_q_invert(b_rgb, reg4, m4):
        b32 = np.asarray(b_rgb, dtype=np.float32)
        inv_b32 = np.float32(1.0) / np.maximum(b32, np.float32(1e-7))
        log10_2 = np.float32(np.log10(2.0))
        min_v = np.float32(1e-7)
        if film == "bw":
            t_arr = np.maximum(rgb4 * inv_b32, min_v)
            gm = np.cbrt(t_arr[..., 0] * t_arr[..., 1] * t_arr[..., 2])
            inp_wp = np.repeat(gm[..., None], 3, axis=-1)
            D = np.repeat((-np.log2(np.maximum(gm, min_v)) * log10_2)[..., None], 3, axis=-1)
            eff_b = (1.0, 1.0, 1.0)
        else:
            inp_wp = rgb4
            D = np.maximum(rgb4 * inv_b32, min_v)
            np.log2(D, out=D)
            D *= -log10_2
            eff_b = tuple(float(x) for x in b_rgb)

        d_hi, wp_info = block_white_point(inp_wp, eff_b, m4, wb4, WHITE_PCT)
        if film != "bw" and WHITE_B_GUARD is not None:
            d_hi, bg_info = blue_white_guard(d_hi, WHITE_B_GUARD, base=eff_b)
            wp_info = dict(wp_info, blue_guard=bg_info)
        if film != "bw" and WHITE_SELF_RATIO is not None:
            d_hi, sr_info = self_ratio_white_point(d_hi, WHITE_SELF_RATIO)
            wp_info = dict(wp_info, self_ratio=sr_info)
        d_lo = (0.0, 0.0, 0.0)
        gamma = 0.6
        lo32 = np.asarray(d_lo, dtype=np.float32)
        hi32 = np.asarray(d_hi, dtype=np.float32)
        inv_span = np.float32(1.0) / np.maximum(hi32 - lo32, np.float32(1e-4))
        # Exact green-channel log_range matching InversionParams.log_range (invert.py)!
        R_leg = float(hi32[1] - lo32[1]) / float(gamma)
        flr_leg = float(10.0 ** (-R_leg))

        ct0, cl0, ch0, cw0 = reg4
        g_p01_core = float(np.percentile(D[ct0:ct0 + ch0, cl0:cl0 + cw0, 1], 1.0))

        # lo32 is (0, 0, 0) -> scale D in-place to save ~34 MB transient allocation
        D *= inv_span
        s = D
        if SPIKE_CLAMP is not None:
            s, _ = clamp_spikes(s, SPIKE_CLAMP, zero_escape=False)
        pos4 = _t18_chunk_from_s(s, R_leg, flr_leg)
        inv_p_obj = InversionParams(base=eff_b, d_lo=d_lo, d_hi=tuple(float(x) for x in d_hi),
                                    gamma=gamma, exposure=1.0, spike_clamp=SPIKE_CLAMP)
        rep = dict(white_pct=WHITE_PCT, white_point=wp_info, range={"G": {"density_p01": g_p01_core}})
        return pos4, inv_p_obj, rep, g_p01_core

    # Early sprocket-backlight guard BEFORE _fast_q_invert to avoid running full inversion twice!
    if (
        base_mode == "auto"
        and "base_rgb" not in extra
        and film == "color"
        and len(base_rgb) == 3
        and base_rgb[1] > 1e-6
        and (base_rgb[0] / base_rgb[1]) < 0.78
        and (base_rgb[2] / base_rgb[1]) > 0.44
    ):
        ct0, cl0, ch0, cw0 = img_reg_4
        g_core = rgb4[ct0:ct0 + ch0:2, cl0:cl0 + cw0:2, 1]
        g_p99_lin = float(np.percentile(g_core, 99.0))
        g_p01_est = float(-np.log10(max(g_p99_lin / max(base_rgb[1], 1e-7), 1e-7)))
        guarded = _sprocket_guard_from_sp(sp, sp_sat, None, base_rgb, g_p01_est, film)
        if guarded is not None:
            base_rgb, base_source = guarded
            base_info = dict(method="auto_sprocket_guard", rgb_render=base_rgb, source=base_source)

    pos4, inv_p, inv_report, _ = _fast_q_invert(base_rgb, img_reg_4, mask4)

    f_down = max(1, -(-max(h4, w4) // max_side))
    if f_down == 2:
        small = _block_mean2_f32(pos4)
    elif f_down > 1:
        small = block_mean(pos4, f_down).astype(np.float32)
    else:
        small = pos4.astype(np.float32)
    hm_small = holder_mask(small)

    # Smart auto frame detection when uncropped and an opaque black film holder (>=4%) is present:
    # replaces fixed central 25% region with the actual inset interior frame so T24 v4 sees full composition.
    if lr_crop is None and hm_small is not None and float(hm_small.mean()) >= 0.04:
        fb = film_box(hm_small)
        if fb is not None:
            sh, sw = small.shape[:2]
            ft, fl, fh_b, fw_b = fb
            area_frac = (fh_b * fw_b) / max(float(sh * sw), 1.0)
            if 0.18 <= area_frac <= 0.92:
                t_n = max(0.0, (ft + 0.04 * fh_b) / sh)
                l_n = max(0.0, (fl + 0.04 * fw_b) / sw)
                b_n = min(1.0, (ft + 0.96 * fh_b) / sh)
                r_n = min(1.0, (fl + 0.96 * fw_b) / sw)
                if (b_n - t_n) >= 0.25 and (r_n - l_n) >= 0.25:
                    auto_box = (t_n, l_n, b_n, r_n)
                    img_reg_full = _lr_crop_to_image_region(auto_box, full_shape)
                    img_reg_4 = _lr_crop_to_image_region(auto_box, (h4, w4))
                    reg_src = "auto_holder_box"
                    mask4 = _build_mask4(img_reg_4)
                    pos4, inv_p, inv_report, _ = _fast_q_invert(base_rgb, img_reg_4, mask4)
                    if f_down == 2:
                        small = _block_mean2_f32(pos4)
                    elif f_down > 1:
                        small = block_mean(pos4, f_down).astype(np.float32)
                    else:
                        small = pos4.astype(np.float32)
                    hm_small = holder_mask(small)

    cfg = _auto_cfg(path, film, engine, preset, raw_m=raw_m)
    rt, rl, rh, rw = img_reg_4
    # pos4 is ALREADY at grid_div (4x) resolution — pass directly at effective 4x scale (avoid 16x double-downsampling!).
    est_f = max(1, 4 // grid_div)
    ei = (pos4, mask4) if est_f == 1 else estimation_input(pos4, img_reg_4, f=est_f)
    auto_info, _, _ = run_auto_estimate(ei, cfg, d_lo=inv_p.d_lo, d_hi=inv_p.d_hi, gamma=inv_p.gamma, film=film)

    msk_small = block_mean(mask4.astype(np.float32), f_down) > 0.5 if f_down > 1 else mask4

    step_p = 2 if (rh * rw) > 500_000 else 1
    roi_s = rgb4[rt:rt + rh:step_p, rl:rl + rw:step_p]
    center_p99 = [float(np.percentile(roi_s[..., c], 99)) for c in range(3)]
    inv_dict = inv_p.to_dict()
    report = dict(inv_report, input_center_p99=center_p99, input_warnings=input_warnings(inv_dict["d_hi"], center_p99))

    pv = Preview(
        path=path, sha256="", factor=f_down * grid_div, positive=small, mask=msk_small,
        image_region=img_reg_full, region_source=reg_src, film=film,
        base_info=base_info, inversion=inv_dict, white_pct=WHITE_PCT, report=report,
        base_inside_image=inside_img, full_shape=full_shape, auto=auto_info,
        load_kwargs=dict(max_side=max_side, auto=cfg, film=film, keep_full=False,
                         base_rgb=base_rgb, base_source=base_source, image_region=img_reg_full),
        full_positive=None, _base_source=base_source, est_input=ei, holder=hm_small,
    )
    return pv, raw_m

def _fused_invert_and_render(
    rgb_lin: np.ndarray,
    *,
    base_rgb,
    d_lo,
    d_hi,
    gamma: float,
    spike_clamp,
    rp: RenderParams | RenderParamsT19 | RenderParamsT24,
    film: str,
    t18_inv: dict | None,
) -> np.ndarray:
    """Chunked multi-threaded float32 inversion + render directly from linear camera/scanner RGB.

    Avoids allocating full-frame float64 density or full-frame intermediate positive arrays.
    """
    h, w = rgb_lin.shape[:2]
    out = np.empty((h, w, 3), dtype=np.float32)
    base32 = np.asarray(base_rgb, dtype=np.float32)
    inv_base32 = np.float32(1.0) / np.maximum(base32, np.float32(1e-7))
    log10_2 = np.float32(np.log10(2.0))
    min_lin = np.float32(1e-7)

    is_t19 = isinstance(rp, T19_FAMILY)
    lo32 = np.asarray(d_lo, dtype=np.float32)
    hi32 = np.asarray(d_hi, dtype=np.float32)
    span32 = np.maximum(hi32 - lo32, np.float32(1e-4))
    inv_span = np.float32(1.0) / span32
    if is_t19:
        spec = t18_inv or dict(version="t23" if isinstance(rp, RenderParamsT24) else "t18", has_cam_matrix=True)
        _, _, rho, dg, d_anchor, eps_arr, R, flr = _t18_affine_params(
            d_lo, d_hi, gamma, has_cam_matrix=bool(spec["has_cam_matrix"]), version=str(spec["version"]),
        )
        inv_rd = (1.0 / (rho * dg)).astype(np.float32)
        anchor32 = d_anchor.astype(np.float32)
        shift32 = (1.0 + eps_arr).astype(np.float32)
        t19_pre = _prepare_t19_lut(rp)
        m_lin_t24 = _t24_linear_matrix(rp) if isinstance(rp, RenderParamsT24) else None
    else:
        R_leg = float(hi32[1] - lo32[1]) / float(gamma)
        flr_leg = float(10.0 ** (-R_leg))

    ranges = [(r0, min(h, r0 + CHUNK_ROWS)) for r0 in range(0, h, CHUNK_ROWS)]

    def _run_chunk(rr):
        r0, r1 = rr
        src_c = rgb_lin[r0:r1]
        if film == "bw":
            # Neutral geometric mean transmittance relative to base_rgb
            t_c = np.maximum(src_c * inv_base32, min_lin)
            gm = np.cbrt(t_c[..., 0] * t_c[..., 1] * t_c[..., 2])
            D_c = -np.log2(np.maximum(gm, min_lin)) * log10_2
            D_c = np.repeat(D_c[..., None], 3, axis=-1)
        else:
            t_c = np.maximum(src_c * inv_base32, min_lin)
            np.log2(t_c, out=t_c)
            t_c *= -log10_2
            D_c = t_c

        if is_t19:
            if spike_clamp is not None:
                s_leg = (D_c - lo32) * inv_span
                s_leg, n_sp = clamp_spikes(s_leg, spike_clamp, zero_escape=False)
                if n_sp > 0:
                    D_c = lo32 + s_leg * span32
            s = (D_c - anchor32) * inv_rd
            s += shift32
            pos_c = _t18_chunk_from_s(s, R, flr)
            if isinstance(rp, RenderParamsT24):
                e_c, _, _ = _render_t24_core(pos_c, rp, m_lin_t24, _t19_pre=t19_pre)
            else:
                e_c, _, _ = _render_t19_core(pos_c, rp, _t19_pre=t19_pre)
            out[r0:r1] = e_c
        else:
            s = (D_c - lo32) * inv_span
            if spike_clamp is not None:
                s, _ = clamp_spikes(s, spike_clamp, zero_escape=False)
            pos_c = _t18_chunk_from_s(s, R_leg, flr_leg)
            e_c, _ = render_image(pos_c, rp, None, chunk_rows=CHUNK_ROWS, fast=True)
            out[r0:r1] = e_c

    if len(ranges) > 2:
        list(_get_pool().map(_run_chunk, ranges))
    else:
        for rr in ranges:
            _run_chunk(rr)
    return out


_EXIF_COPY_TAGS = (
    "-Make", "-Model", "-LensModel", "-LensInfo", "-FocalLength",
    "-DateTimeOriginal", "-CreateDate", "-ModifyDate",
    "-ISO", "-FNumber", "-ExposureTime", "-Artist", "-Copyright",
    "-GPS:All",
)


def _safe_exif_path(p: Path) -> str | None:
    s = str(Path(p).resolve())
    if "\n" in s or "\r" in s or "\x00" in s or s.startswith("-"):
        return None
    return s


class _ExifToolDaemon:
    """Thread-safe persistent exiftool (-stay_open True) worker: cuts EXIF copy from ~1.5s to ~0.05s."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._seq = 0
        self._tool: str | None = None

    def _close_locked(self, *, kill: bool = False) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if not kill and proc.poll() is None and proc.stdin is not None:
                proc.stdin.write("-stay_open\nFalse\n")
                proc.stdin.flush()
                proc.wait(timeout=2)
                return
        except Exception:  # noqa: BLE001
            pass
        try:
            if proc.poll() is None:
                proc.kill() if kill else proc.terminate()
                proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    def _ensure_locked(self, tool: str) -> subprocess.Popen | None:
        if self._proc is not None and self._proc.poll() is None and self._tool == tool:
            return self._proc
        self._close_locked()
        try:
            kw_win = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if sys.platform == "win32" else {}
            self._proc = subprocess.Popen(
                [tool, "-stay_open", "True", "-@", "-"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1, **kw_win,
            )
            self._tool = tool
            return self._proc
        except OSError:
            self._proc = None
            return None

    @staticmethod
    def _readline_with_deadline(proc: subprocess.Popen, deadline: float) -> str | None:
        assert proc.stdout is not None
        rem = deadline - time.monotonic()
        if rem <= 0:
            return None
        if sys.platform != "win32":
            ready, _, _ = select.select([proc.stdout], [], [], rem)
            if not ready:
                return None
            return proc.stdout.readline()
        out: list[str] = []
        def _rd():
            try:
                out.append(proc.stdout.readline())
            except Exception:  # noqa: BLE001
                pass
        t = threading.Thread(target=_rd, daemon=True)
        t.start()
        t.join(rem)
        if t.is_alive() or not out:
            return None
        return out[0]

    def copy_tags(self, tool: str, src: Path, dst: Path, timeout_s: float = 15.0) -> bool:
        src_s = _safe_exif_path(src)
        dst_s = _safe_exif_path(dst)
        if not src_s or not dst_s:
            return False
        with self._lock:
            proc = self._ensure_locked(tool)
            if proc is None or proc.stdin is None or proc.stdout is None:
                return False
            self._seq += 1
            marker = f"{{ready{self._seq}}}"
            args = [
                "-overwrite_original", "-q", "-q",
                "-TagsFromFile", src_s,
                *_EXIF_COPY_TAGS,
                "--",
                dst_s,
                f"-execute{self._seq}",
            ]
            deadline = time.monotonic() + float(timeout_s)
            try:
                proc.stdin.write("\n".join(args) + "\n")
                proc.stdin.flush()
                while True:
                    line = self._readline_with_deadline(proc, deadline)
                    if not line:
                        self._close_locked(kill=True)
                        return False
                    if line.strip() == marker:
                        return True
            except Exception:  # noqa: BLE001
                self._close_locked(kill=True)
                return False

    def close(self) -> None:
        with self._lock:
            self._close_locked()


_EXIFTOOL_DAEMON = _ExifToolDaemon()
atexit.register(_EXIFTOOL_DAEMON.close)


def _copy_exif_metadata(src: Path, dst: Path) -> bool:
    tool = _exiftool()
    if not tool:
        return False
    src_s = _safe_exif_path(src)
    dst_s = _safe_exif_path(dst)
    if not src_s or not dst_s:
        return False
    if _EXIFTOOL_DAEMON.copy_tags(tool, src, dst):
        return True
    try:
        kw_win = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if sys.platform == "win32" else {}
        proc = subprocess.run(
            [tool, "-overwrite_original", "-q", "-q", "-TagsFromFile", src_s, *_EXIF_COPY_TAGS, "--", dst_s],
            capture_output=True, text=True, check=False, timeout=15, **kw_win,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


class PluginBridgeState:
    def __init__(self, *, allow_frozen: bool = True, max_preview_side: int = PREVIEW_MAX_SIDE):
        self.allow_frozen = allow_frozen
        self.max_preview_side = max_preview_side
        self.lock = threading.Lock()
        self._open_cond = threading.Condition(self.lock)
        self._loading_key: tuple | None = None
        self.convert_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_requests = 0
        self._active_idle = threading.Condition(self._active_lock)
        self.last_activity = time.monotonic()

        self.cache_dir = _cache_root() / "plugin_bridge"
        self.preview_dir = self.cache_dir / "previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self._preview_seq = 0
        self._recent_previews: list[Path] = []
        self._recent_neg_previews: list[Path] = []

        self._open_key: tuple | None = None
        self._open_pv: Preview | None = None
        self._open_raw_m = None
        self._open_neg_path: Path | None = None
        self._open_neg_size: tuple | None = None
        self._open_neg_u8: np.ndarray | None = None
        self._preview_proxy_key: tuple | None = None
        self._preview_proxy: tuple | None = None

        self.rolls: dict[str, dict] = {}
        self._purge_stale_previews()

    @property
    def active_requests(self) -> int:
        with self._active_lock:
            return self._active_requests

    @contextmanager
    def track_request(self):
        with self._active_lock:
            self._active_requests += 1
        self.touch()
        try:
            yield
        finally:
            self.touch()
            with self._active_lock:
                self._active_requests = max(0, self._active_requests - 1)
                if self._active_requests == 0:
                    self._active_idle.notify_all()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + float(timeout)
        with self._active_lock:
            while self._active_requests > 0:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return False
                self._active_idle.wait(timeout=rem)
            return True

    def _purge_stale_previews(self, max_age_s: float = 3600.0) -> None:
        now = time.time()
        try:
            for f in self.preview_dir.iterdir():
                if not f.is_file():
                    continue
                is_preview_png = f.suffix == ".png" and (f.name.startswith("prev_") or f.name.startswith("neg_"))
                is_stale_tmp = f.name.startswith(".") and f.name.endswith(".tmp")
                if is_preview_png or is_stale_tmp:
                    if now - f.stat().st_mtime > max_age_s:
                        f.unlink(missing_ok=True)
        except OSError:
            pass

    def trim_caches(self, *, keep_displayed: bool = True) -> None:
        """Release cached preview/roll arrays when bridge has been idle to minimize resident RAM."""
        with self.lock:
            had = self._open_pv is not None or self._open_neg_u8 is not None or bool(self.rolls)
            self._open_pv = None
            self._open_key = None
            self._open_raw_m = None
            self._open_neg_u8 = None
            self._open_neg_size = None
            if not keep_displayed:
                self._open_neg_path = None
            self._preview_proxy_key = None
            self._preview_proxy = None
            self.rolls.clear()
            self._cleanup_old_previews_locked(keep=1 if keep_displayed else 0)
            self._cleanup_old_neg_previews_locked(keep=1 if keep_displayed else 0)
        if had:
            gc.collect()

    def touch(self):
        self.last_activity = time.monotonic()

    def _cleanup_old_previews_locked(self, keep: int = 4):
        while len(self._recent_previews) > keep:
            old = self._recent_previews.pop(0)
            try:
                old.unlink(missing_ok=True)
            except OSError:
                pass

    def _cleanup_old_neg_previews_locked(self, keep: int = 4):
        while len(self._recent_neg_previews) > keep:
            old = self._recent_neg_previews.pop(0)
            if old == self._open_neg_path and keep > 0:
                self._recent_neg_previews.append(old)
                break
            try:
                old.unlink(missing_ok=True)
            except OSError:
                pass

    def _write_preview_file(self, img_u8: np.ndarray, *, prefix: str = "prev", track_recent: bool = True) -> Path:
        raw_png = png_bytes(img_u8, compress_level=0)
        with self.lock:
            self._preview_seq += 1
            seq = self._preview_seq
        out = self.preview_dir / f"{prefix}_{os.getpid()}_{seq}.png"
        tmp = self.preview_dir / f".{out.name}.tmp"
        tmp.write_bytes(raw_png)
        os.replace(tmp, out)
        with self.lock:
            if track_recent:
                self._recent_previews.append(out)
                self._cleanup_old_previews_locked(keep=6)
            elif prefix == "neg":
                self._recent_neg_previews.append(out)
                self._cleanup_old_neg_previews_locked(keep=4)
        return out

    def _roll_base_for(self, path_str: str, base_mode: str, roll_id: str | None, softness: float = 0.5) -> dict:
        if not roll_id or base_mode == "auto":
            return {}
        rinfo = self.rolls.get(roll_id)
        if not rinfo:
            return {}
        fe = rinfo["frames"].get(str(Path(path_str).resolve()))
        if not fe or fe.get("roll_rgb") is None:
            return {}
        if base_mode == "roll_nat":
            b_nat, nat_diag = natural_roll_base(
                fe["roll_rgb"], fe["block48_L_p01"], own_rgb=fe["own_rgb"], softness=float(softness)
            )
            return dict(base_rgb=list(b_nat), base_source=dict(method="roll_nat", roll_rgb=fe["roll_rgb"],
                                                               own_rgb=fe["own_rgb"], natural_info=nat_diag))
        if base_mode == "roll_dq" and fe.get("dq_rgb") is not None:
            return dict(base_rgb=list(fe["dq_rgb"]), base_source=dict(method="roll_dq", roll_rgb=fe["roll_rgb"]))
        if base_mode == "roll_b1":
            return dict(base_rgb=list(fe["roll_rgb"]), base_source=dict(method="roll_b1", roll_rgb=fe["roll_rgb"]))
        return {}

    def _make_open_key(self, resolved: Path, film: str, base_mode: str, roll_id: str | None,
                       softness: float, lr_crop: tuple[float, float, float, float] | None) -> tuple:
        st = resolved.stat()
        crop_key = tuple(round(x, 4) for x in lr_crop) if lr_crop else None
        return (str(resolved), st.st_size, st.st_mtime_ns, film, base_mode, roll_id or "",
                round(float(softness), 2), crop_key)

    def _ensure_open(self, path: Path, film: str, engine: str, preset: str,
                     base_mode: str = "auto", roll_id: str | None = None, softness: float = 0.5,
                     lr_crop: tuple[float, float, float, float] | None = None) -> tuple[Preview, list | None, tuple]:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"source file not found: {path}")
        base_key = self._make_open_key(resolved, film, base_mode, roll_id, softness, lr_crop)
        want_cfg = NOREF_T25["id"] if engine == "t25" else (NOREF_T24["id"] if engine == "t24" else (NOREF_T23["id"] if engine == "t23" else NOREF_V3["id"]))

        while True:
            with self.lock:
                if self._open_pv is not None and self._open_key == base_key:
                    pv = self._open_pv
                    raw_m = self._open_raw_m
                    cur_cfg = (pv.auto or {}).get("config", "")
                    cur_preset = (pv.auto or {}).get("preset")
                    if cur_cfg == want_cfg and not (engine == "v3" and cur_preset != preset):
                        return pv, raw_m, base_key
                    # Re-estimate auto config without holding lock across CPU work
                    if self._loading_key == base_key:
                        self._open_cond.wait(timeout=0.5)
                        continue
                    self._loading_key = base_key
                    re_auto_pv, re_auto_rm = pv, raw_m
                elif self._loading_key == base_key:
                    self._open_cond.wait(timeout=0.5)
                    continue
                else:
                    re_auto_pv = None
                    re_auto_rm = None
                    self._loading_key = base_key
                    extra = self._roll_base_for(str(resolved), base_mode, roll_id, softness)

            if re_auto_pv is not None:
                try:
                    cfg = _auto_cfg(resolved, film, engine, preset, raw_m=re_auto_rm)
                    updated_pv = re_auto_pv.with_auto(cfg)
                    with self.lock:
                        if self._open_key == base_key:
                            self._open_pv = updated_pv
                        if self._loading_key == base_key:
                            self._loading_key = None
                        self._open_cond.notify_all()
                    return updated_pv, re_auto_rm, base_key
                except Exception:
                    with self.lock:
                        if self._loading_key == base_key:
                            self._loading_key = None
                        self._open_cond.notify_all()
                    raise

            # Heavy RAW/scanner decode runs completely outside self.lock
            try:
                if not is_scanner_dng(resolved, allow_frozen=True):
                    pv, raw_m = _fast_load_camera_preview(
                        resolved, max_side=self.max_preview_side, film=film, engine=engine, preset=preset,
                        base_mode=base_mode, extra=extra, lr_crop=lr_crop,
                    )
                else:
                    if lr_crop is not None:
                        extra["image_region"] = _lr_crop_to_image_region(lr_crop, _scanner_shape(resolved))
                    cfg = _auto_cfg(resolved, film, engine, preset)
                    wk = dict(white_block=WHITE_BLOCK, white_pct=WHITE_PCT, white_b_guard=WHITE_B_GUARD,
                              white_self_ratio=WHITE_SELF_RATIO, spike_clamp=SPIKE_CLAMP)
                    pv = load_preview(resolved, max_side=self.max_preview_side, film=film, keep_full=False, auto=cfg, **wk, **extra)
                    raw_m = None
                with self.lock:
                    if self._open_key != base_key:
                        self._open_neg_path = None
                        self._open_neg_size = None
                        self._open_neg_u8 = None
                        self._preview_proxy_key = None
                        self._preview_proxy = None
                    self._open_pv = pv
                    self._open_raw_m = raw_m
                    self._open_key = base_key
                    if self._loading_key == base_key:
                        self._loading_key = None
                    self._open_cond.notify_all()
                return pv, raw_m, base_key
            except Exception:
                with self.lock:
                    if self._loading_key == base_key:
                        self._loading_key = None
                    self._open_cond.notify_all()
                raise

    def _effective_auto_for_preview(self, pv: Preview, engine: str, preset: str,
                                    roll_id: str | None = None, use_roll_pool: bool = False,
                                    raw_m=None) -> tuple[dict, dict]:
        raw_p = dict(pv.auto["params"])
        if use_roll_pool and roll_id and engine in T25_ENGINES:
            rinfo = self.rolls.get(roll_id)
            pooled = (rinfo or {}).get("pooled_t24", {}).get(str(pv.path.resolve()))
            if pooled:
                raw_p = dict(pooled)
        if engine in ("t25", "t24", "t23"):
            rm = raw_m if raw_m is not None else (_read_raw_xyz_matrix(pv.path) if pv.film == "color" else None)
            sm = _scene_metrics_for_preview(pv)
            return _apply_t24_preset(raw_p, preset, pv.film, engine, raw_matrix=rm, scene_metrics=sm)
        return raw_p, {}

    def preview(self, body: dict) -> dict:
        self.touch()
        if not body.get("path"):
            raise ValueError("missing required field 'path'")
        path = Path(body["path"])
        film = body.get("film", "color")
        if film not in FILM_MODES:
            raise ValueError(f"film must be one of {FILM_MODES}")
        engine = body.get("engine", "t25")
        preset = body.get("preset", "standard")
        base_mode = body.get("base_mode", "auto")
        roll_id = body.get("roll_id")
        raw_soft = float(body.get("softness", 0.5))
        softness = float(np.clip(raw_soft if np.isfinite(raw_soft) else 0.5, 0.0, 1.0))
        use_roll_pool = bool(body.get("roll_pool", False))
        auto_crop = bool(body.get("auto_crop", False))
        use_lr_crop = bool(body.get("use_lr_crop", True))
        lr_crop = _normalize_lr_crop(body.get("lr_crop")) if use_lr_crop else None
        lr_orient = body.get("lr_orientation")
        lr_rot90 = LR_ORIENT_TO_ROT90.get(str(lr_orient), 0) if (lr_orient and use_lr_crop) else 0
        lr_flip_h = (str(lr_orient) in LR_ORIENT_FLIP_H) if (lr_orient and use_lr_crop) else False
        rot_delta = int(body.get("rot_delta", 0) or 0) % 4
        total_rot90 = (int(body["rot90"]) % 4) if "rot90" in body else ((lr_rot90 + rot_delta) % 4)
        raw_ang = float(body.get("lr_angle", 0.0) or 0.0) if use_lr_crop else 0.0
        lr_angle = raw_ang if np.isfinite(raw_ang) else 0.0
        view = body.get("view", "after")
        want_b64 = bool(body.get("return_base64", False))
        adj = body.get("adjustments") or {}

        max_w = _parse_max_dim(body.get("max_width"))
        max_h = _parse_max_dim(body.get("max_height"))
        need_disk_file = (not want_b64) or (body.get("write_preview_file") is not False)

        _gerr = body.get("lr_geom_err")
        print(f"[preview] {path.name} engine={engine} preset={preset} use_lr_crop={use_lr_crop} lr_crop={lr_crop} orient={lr_orient} angle={lr_angle:.2f}" + (f" geom_err={_gerr}" if _gerr else ""), flush=True)
        pv, raw_m, open_key = self._ensure_open(path, film, engine, preset, base_mode, roll_id, softness, lr_crop=lr_crop)
        has_border = _preview_has_border(pv)

        if view == "negative":
            crop_neg = lr_crop if lr_crop is not None else (_auto_border_crop_norm(pv) if auto_crop else None)
            neg_size = (open_key, max_w, max_h, total_rot90, lr_flip_h, crop_neg, round(lr_angle, 2))
            with self.lock:
                cached_neg_path = self._open_neg_path if self._open_neg_size == neg_size else None
                cached_neg_u8 = self._open_neg_u8 if self._open_neg_size == neg_size else None
            neg_u8 = cached_neg_u8
            out_path = cached_neg_path if (cached_neg_path is not None and cached_neg_path.exists()) else None
            if neg_u8 is None or (need_disk_file and out_path is None):
                from PIL import Image
                raw_bytes, _ = negative_view(pv.path, max_side=self.max_preview_side, expected_shape=tuple(pv.full_shape))
                im = Image.open(_io.BytesIO(raw_bytes)).convert("RGB")
                arr_n = np.asarray(im, dtype=np.float32) * np.float32(1.0 / 255.0)
                if crop_neg is not None or abs(lr_angle) >= 0.05:
                    arr_n = apply_lr_crop_angle(arr_n, crop_neg, lr_angle, fast_preview=True)
                if total_rot90:
                    arr_n = np.ascontiguousarray(np.rot90(arr_n, total_rot90))
                if lr_flip_h:
                    arr_n = np.ascontiguousarray(arr_n[:, ::-1])
                im = Image.fromarray(to8(arr_n), mode="RGB")
                if max_w or max_h:
                    mw = max_w if max_w else im.width
                    mh = max_h if max_h else im.height
                    scale = min(mw / max(im.width, 1), mh / max(im.height, 1), 1.0)
                    if scale < 0.999:
                        im = im.resize((max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale)))), Image.Resampling.LANCZOS)
                neg_u8 = np.asarray(im, dtype=np.uint8)
                if need_disk_file:
                    out_path = self._write_preview_file(neg_u8, prefix="neg", track_recent=False)
                with self.lock:
                    if self._open_key == open_key:
                        if out_path is not None:
                            self._open_neg_path = out_path
                        self._open_neg_u8 = neg_u8
                        self._open_neg_size = neg_size
            res = dict(
                preview_path=str(out_path) if out_path is not None else None,
                width=int(neg_u8.shape[1]), height=int(neg_u8.shape[0]),
                engine=engine, preset=preset, film=film, view=view,
                base_method=(pv._base_source or {}).get("method") or pv.base_info.get("method"),
                full_shape=list(pv.full_shape),
                has_border=has_border,
                warnings=[w["message"] for w in pv.report.get("input_warnings", [])],
            )
            if want_b64:
                res["image_data_url"] = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes(neg_u8, quality=90)).decode("ascii")
            return res

        if view == "auto":
            rp = coerce_render_params(dict(pv.auto["params"]))
            grade = DarkroomGrade()
        else:
            eff_p, eff_g = self._effective_auto_for_preview(pv, engine, preset, roll_id, use_roll_pool, raw_m=raw_m)
            final_p, grade = _apply_user_adjustments(eff_p, eff_g, adj, engine=engine, film=film)
            rp = coerce_render_params(final_p)

        crop_box = lr_crop if lr_crop is not None else (_auto_border_crop_norm(pv) if auto_crop else None)
        is_t19_fam = isinstance(rp, (RenderParamsT19, RenderParamsT24))

        if max_w or max_h:
            # Build/reuse display-resolution linear preview proxy so slider drags render only target pixels
            t18_spec_key = (bool(pv.t18_inversion()["has_cam_matrix"]), str(pv.t18_inversion()["version"])) if is_t19_fam else None
            pkey = (open_key, is_t19_fam, t18_spec_key, crop_box, round(lr_angle, 2), total_rot90, lr_flip_h, max_w, max_h)
            with self.lock:
                proxy_hit = self._preview_proxy if self._preview_proxy_key == pkey else None
            if proxy_hit is not None:
                p_lin, p_mask, p_hmask, bb_share = proxy_hit
            else:
                if is_t19_fam:
                    src_lin, bb_share = pv._t18_positive()
                else:
                    src_lin = pv.positive
                    m0 = pv.mask
                    n0 = int(m0.sum())
                    bb_share = float((np.minimum(np.minimum(src_lin[..., 0], src_lin[..., 1]), src_lin[..., 2]) < 0)[m0].sum() / n0) if n0 else 0.0
                hmask0 = pv._get_holder_paint_mask()
                m_cur = pv.mask
                if crop_box is not None or abs(lr_angle) >= 0.05:
                    src_lin = apply_lr_crop_angle(src_lin, crop_box, lr_angle, fast_preview=True)
                    m_cur = apply_lr_crop_angle(m_cur.astype(np.float32)[..., None], crop_box, lr_angle, fast_preview=True)[..., 0] > 0.5
                    if hmask0 is not None:
                        hmask0 = apply_lr_crop_angle(hmask0.astype(np.float32)[..., None], crop_box, lr_angle, fast_preview=True)[..., 0] > 0.5
                if total_rot90:
                    src_lin = np.ascontiguousarray(np.rot90(src_lin, total_rot90))
                    m_cur = np.ascontiguousarray(np.rot90(m_cur, total_rot90))
                    if hmask0 is not None:
                        hmask0 = np.ascontiguousarray(np.rot90(hmask0, total_rot90))
                if lr_flip_h:
                    src_lin = np.ascontiguousarray(src_lin[:, ::-1])
                    m_cur = np.ascontiguousarray(m_cur[:, ::-1])
                    if hmask0 is not None:
                        hmask0 = np.ascontiguousarray(hmask0[:, ::-1])
                ih, iw = src_lin.shape[:2]
                mw = max_w if max_w else iw
                mh = max_h if max_h else ih
                scale = min(mw / max(iw, 1), mh / max(ih, 1), 1.0)
                if scale < 0.999:
                    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
                    p_lin = _resize_linear_f32(src_lin, nw, nh)
                    p_mask = _resize_mask_nn(m_cur, nw, nh)
                    p_hmask = _resize_mask_nn(hmask0, nw, nh) if hmask0 is not None else None
                else:
                    p_lin, p_mask, p_hmask = src_lin, m_cur, hmask0
                with self.lock:
                    if self._open_key == open_key:
                        self._preview_proxy_key = pkey
                        self._preview_proxy = (p_lin, p_mask, p_hmask, bb_share)

            if isinstance(rp, RenderParamsT24):
                disp, diag = render_t24(p_lin, rp, p_mask)
                aw_share = float(diag["above_white"])
            elif isinstance(rp, RenderParamsT19):
                disp, diag = render_t19(p_lin, rp, p_mask)
                aw_share = float(diag["above_white"])
            else:
                from .render import render_fast_full
                disp, _, _, pre = render_fast_full(p_lin, rp)
                nm = int(p_mask.sum())
                aw_share = float((np.maximum(np.maximum(pre[..., 0], pre[..., 1]), pre[..., 2]) > 1)[p_mask].sum() / nm) if nm else 0.0
            if p_hmask is not None:
                from .preview import HOLDER_GREY
                disp[p_hmask] = np.float32(HOLDER_GREY)
            stats = dict(below_black=bb_share, above_white=aw_share)
            if not grade.is_identity():
                disp = apply_darkroom_grade(disp, grade, film=film, apply_vignette_here=False, fast=True)
            if grade.vignette:
                disp = apply_vignette(disp, grade.vignette, inplace=True)
            u8 = to8(disp)
        else:
            disp, stats = pv.render_stats(rp)
            if not grade.is_identity():
                disp = apply_darkroom_grade(disp, grade, film=film, apply_vignette_here=False, fast=True)
            if crop_box is not None or abs(lr_angle) >= 0.05:
                disp = apply_lr_crop_angle(disp, crop_box, lr_angle, fast_preview=True)
            if total_rot90:
                disp = np.ascontiguousarray(np.rot90(disp, total_rot90))
            if lr_flip_h:
                disp = np.ascontiguousarray(disp[:, ::-1])
            if grade.vignette:
                disp = apply_vignette(disp, grade.vignette, inplace=True)
            u8 = to8(disp)
        out_path = self._write_preview_file(u8, prefix="prev") if need_disk_file else None
        bb_val = float(stats["below_black"])
        aw_val = float(stats["above_white"])
        res = dict(
            preview_path=str(out_path) if out_path is not None else None,
            width=int(u8.shape[1]), height=int(u8.shape[0]),
            engine=engine, preset=preset, film=film, view=view,
            base_method=(pv._base_source or {}).get("method") or pv.base_info.get("method"),
            full_shape=list(pv.full_shape),
            has_border=has_border,
            below_black=bb_val if math.isfinite(bb_val) else 0.0,
            above_white=aw_val if math.isfinite(aw_val) else 0.0,
            warnings=[w["message"] for w in pv.report.get("input_warnings", [])],
        )
        if want_b64:
            res["image_data_url"] = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes(u8, quality=90)).decode("ascii")
        return res

    def roll_analyze(self, body: dict) -> dict:
        self.touch()
        raw_paths = [Path(p).resolve() for p in (body.get("paths") or [])]
        if not raw_paths:
            raise ValueError("roll_analyze requires non-empty 'paths'")
        film = body.get("film", "color")
        if film not in FILM_MODES:
            raise ValueError(f"film must be one of {FILM_MODES}")
        engine = body.get("engine", "t25")
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
        if len(raw_paths) > 500:
            raise ValueError(f"too many frames in a single roll_analyze call ({len(raw_paths)} > 500)")
        do_pool = bool(body.get("roll_pool", True)) and engine in T25_ENGINES and len(raw_paths) >= 2

        from concurrent.futures import ThreadPoolExecutor

        exif = _exif_info(raw_paths)
        existing_paths = [p for p in raw_paths if p.is_file()]
        own_map, exp_map, scans, preloaded_map = {}, {}, {}, {}
        keep_preload = len(existing_paths) <= 40

        def _scan_frame(p: Path):
            self.touch()
            is_scan = is_scanner_dng(p, allow_frozen=True)
            preload = None
            if is_scan:
                fr = load_scanner(p, allow_frozen=True)
                sp, sp_sat = block_means(fr.rgb, fr.sat, 24)
                del fr
                blk = 24
            else:
                with open_rawpy(p) as r:
                    raw_m = np.asarray(r.rgb_xyz_matrix)[:3].tolist()
                    s_sz = r.sizes
                    full_shape = (int(s_sz.height), int(s_sz.width))
                    img_u16 = r.postprocess(
                        half_size=True, output_bps=16, gamma=(1, 1),
                        no_auto_bright=True, no_auto_scale=False,
                        use_camera_wb=False, use_auto_wb=False, user_wb=[1, 1, 1, 1],
                        output_color=rawpy.ColorSpace.raw, user_flip=0,
                    )
                hh_r, hw_r = img_u16.shape[:2]
                h2_r, w2_r = (hh_r // 2) * 2, (hw_r // 2) * 2
                sat_half = (img_u16[..., 0] >= 65470) | (img_u16[..., 1] >= 65470) | (img_u16[..., 2] >= 65470)
                sat_4 = (
                    sat_half[0:h2_r:2, 0:w2_r:2]
                    | sat_half[0:h2_r:2, 1:w2_r:2]
                    | sat_half[1:h2_r:2, 0:w2_r:2]
                    | sat_half[1:h2_r:2, 1:w2_r:2]
                )
                del sat_half
                s_u32 = img_u16[0:h2_r:2, 0:w2_r:2].astype(np.uint32)
                s_u32 += img_u16[0:h2_r:2, 1:w2_r:2]
                s_u32 += img_u16[1:h2_r:2, 0:w2_r:2]
                s_u32 += img_u16[1:h2_r:2, 1:w2_r:2]
                del img_u16
                rgb4 = s_u32.astype(np.float32)
                del s_u32
                rgb4 *= np.float32(0.25 / 65535.0)
                sp, sp_sat = block_means(rgb4, sat_4, 6)
                if do_pool and keep_preload:
                    preload = (rgb4, sat_4, 4, raw_m, full_shape, sp, sp_sat)
                else:
                    del rgb4, sat_4
                blk = 12
            p01 = block48_L_p01(sp, sp_sat, block=blk, crop=None)
            try:
                est = estimate_base_blocks(sp, sp_sat, block=blk, crop=None, exclude_near_saturated=not is_scan)
                own = list(est.rgb)
            except ValueError:
                own = None
            return p, own, p01, preload

        if len(existing_paths) >= 2:
            with ThreadPoolExecutor(max_workers=min(3, len(existing_paths))) as ex:
                scan_results = list(ex.map(_scan_frame, existing_paths))
        else:
            scan_results = [_scan_frame(p) for p in existing_paths]

        for p, own, p01, preload in scan_results:
            ps = str(p)
            exp_v = exif[p]["exposure"]
            if own is not None:
                own_map[ps] = own
                exp_map[ps] = exp_v
            scans[ps] = dict(own_rgb=own, block48_L_p01=p01, exposure=exp_v)
            if preload is not None:
                preloaded_map[ps] = preload

        rb = None
        if len(own_map) >= 2:
            try:
                rb = estimate_base_roll(own_map, exp_map)
            except ValueError:
                rb = None

        frames_out = {}
        for p_str, sc in scans.items():
            entry = dict(own_rgb=sc["own_rgb"], block48_L_p01=sc["block48_L_p01"], roll_rgb=None, dq_rgb=None, nat_rgb=None)
            if rb is not None:
                b_roll = list(rb.for_exposure(sc["exposure"]))
                b_dq, _ = decoupled_quantile_base(b_roll, sc["block48_L_p01"])
                b_nat, _ = natural_roll_base(b_roll, sc["block48_L_p01"], own_rgb=sc["own_rgb"], softness=0.5)
                dev = base_deviation(sc["own_rgb"], b_roll) if sc["own_rgb"] is not None else None
                entry.update(roll_rgb=b_roll, dq_rgb=list(b_dq), nat_rgb=list(b_nat),
                             chroma_part=dev["chroma_part"] if dev else None)
            frames_out[p_str] = entry

        raw_lr_crops = body.get("lr_crops") or {}
        norm_crops: dict[str, tuple[float, float, float, float]] = {}
        if isinstance(raw_lr_crops, dict):
            for kp, kc in raw_lr_crops.items():
                nc = _normalize_lr_crop(kc)
                if nc is not None:
                    norm_crops[str(Path(kp).resolve())] = nc

        # Fast T21b roll pooling using half-size camera preview (respecting per-photo lr_crop!)
        pooled_t24: dict[str, dict] = {}
        try:
          if do_pool and len(raw_paths) >= 2:
            per_params, valid_paths = [], []
            for p in raw_paths:
                if not p.is_file():
                    continue
                self.touch()
                try:
                    fe = frames_out.get(str(p), {})
                    extra = dict(base_rgb=fe["nat_rgb"]) if fe.get("nat_rgb") is not None else {}
                    p_crop = norm_crops.get(str(p))
                    pl = preloaded_map.pop(str(p), None)
                    if pl is not None or not is_scanner_dng(p, allow_frozen=True):
                        pv, _ = _fast_load_camera_preview(
                            p, max_side=256, film=film, engine=engine, preset="standard",
                            base_mode="roll_nat" if extra else "auto", extra=extra, lr_crop=p_crop,
                            preloaded=pl,
                        )
                    else:
                        if p_crop is not None:
                            extra["image_region"] = _lr_crop_to_image_region(p_crop, _scanner_shape(p))
                        cfg = _auto_cfg(p, film, engine, "standard")
                        wk = dict(white_block=WHITE_BLOCK, white_pct=WHITE_PCT, white_b_guard=WHITE_B_GUARD,
                                  white_self_ratio=WHITE_SELF_RATIO, spike_clamp=SPIKE_CLAMP)
                        pv = load_preview(p, max_side=256, film=film, keep_full=False, auto=cfg, **wk, **extra)
                    per_params.append(pv.auto["params"])
                    valid_paths.append(str(p))
                    del pv
                except Exception:  # noqa: BLE001
                    pass
            if len(per_params) >= 2:
                pooled_list = pool_roll_params_t24(per_params)
                for ps, pp in zip(valid_paths, pooled_list):
                    pooled_t24[ps] = pp
        finally:
            preloaded_map.clear()

        roll_id = secrets.token_hex(6)
        with self.lock:
            self.rolls[roll_id] = dict(
                roll_id=roll_id, n_files=len(scans), has_roll_base=rb is not None,
                frames=frames_out, pooled_t24=pooled_t24,
            )
            if len(self.rolls) > 8:
                del self.rolls[next(iter(self.rolls))]
        return dict(roll_id=roll_id, n_files=len(scans), has_roll_base=rb is not None, n_pooled=len(pooled_t24))

    def convert_one(self, body: dict) -> dict:
        """Single-pass, low-RAM full-resolution FilmColor conversion (reuses preview estimate when available)."""
        self.touch()
        if not body.get("path"):
            raise ValueError("missing required field 'path'")
        src = Path(body["path"]).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"source file not found: {src}")
        film = body.get("film", "color")
        if film not in FILM_MODES:
            raise ValueError(f"film must be one of {FILM_MODES}")
        engine = body.get("engine", "t25")
        preset = body.get("preset", "standard")
        base_mode = body.get("base_mode", "auto")
        roll_id = body.get("roll_id")
        use_roll_pool = bool(body.get("roll_pool", False))
        raw_soft_c = float(body.get("softness") if body.get("softness") is not None else 0.5)
        softness = float(np.clip(raw_soft_c if np.isfinite(raw_soft_c) else 0.5, 0.0, 1.0))
        auto_crop = bool(body.get("auto_crop", False))
        use_lr_crop = bool(body.get("use_lr_crop", True))
        lr_crop = _normalize_lr_crop(body.get("lr_crop")) if use_lr_crop else None
        lr_orient = body.get("lr_orientation")
        lr_rot90 = LR_ORIENT_TO_ROT90.get(str(lr_orient), 0) if (lr_orient and use_lr_crop) else 0
        lr_flip_h = (str(lr_orient) in LR_ORIENT_FLIP_H) if (lr_orient and use_lr_crop) else False
        rot_delta = int(body.get("rot_delta", 0) or 0) % 4
        rot90 = (int(body["rot90"]) % 4) if "rot90" in body else ((lr_rot90 + rot_delta) % 4)
        raw_ang_c = float(body.get("lr_angle", 0.0) or 0.0) if use_lr_crop else 0.0
        lr_angle = raw_ang_c if np.isfinite(raw_ang_c) else 0.0
        fmt = str(body.get("format", "tiff16")).lower()
        if fmt not in ("tiff16", "jpeg", "png"):
            raise ValueError(f"unsupported format {fmt!r}; expected 'tiff16', 'jpeg', or 'png'")
        encoding = str(body.get("encoding", "srgb")).lower()
        if encoding not in ("srgb", "linear"):
            raise ValueError(f"unsupported encoding {encoding!r}; expected 'srgb' or 'linear'")
        raw_q = body.get("quality")
        quality = int(np.clip(int(95 if raw_q is None else raw_q), 50, 100))
        max_edge = body.get("max_edge")
        if max_edge not in (None, "", 0, "0"):
            max_edge = int(max_edge)
            if not (64 <= max_edge <= 65535):
                raise ValueError(f"max_edge must be in [64, 65535], got {max_edge}")
        else:
            max_edge = None
        write_sidecar = bool(body.get("write_sidecar", False))
        copy_exif = bool(body.get("copy_exif", True))
        suffix = str(body.get("suffix", "_FilmColor"))
        if "/" in suffix or "\\" in suffix or ".." in suffix:
            raise ValueError(f"invalid suffix: {suffix!r}")
        adj = body.get("adjustments") or {}

        out_dir_raw = body.get("out_dir")
        if out_dir_raw:
            out_dir = Path(out_dir_raw).resolve()
        elif body.get("subfolder"):
            sub_p = Path(str(body["subfolder"]))
            if sub_p.is_absolute() or ".." in sub_p.parts:
                raise ValueError(f"invalid subfolder: {body['subfolder']!r}")
            out_dir = (src.parent / sub_p).resolve()
        else:
            out_dir = src.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        _gerr = body.get("lr_geom_err")
        print(f"[convert] {src.name} engine={engine} preset={preset} use_lr_crop={use_lr_crop} lr_crop={lr_crop} orient={lr_orient} angle={lr_angle:.2f}" + (f" geom_err={_gerr}" if _gerr else ""), flush=True)
        with self.convert_lock:
            self.touch()
            # Step 1: Obtain estimation (0.00s if already cached from preview; else fast half-size pass!)
            pv, raw_m, _ = self._ensure_open(src, film, engine, preset, base_mode, roll_id, softness, lr_crop=lr_crop)
            with self.lock:
                eff_p, eff_g = self._effective_auto_for_preview(pv, engine, preset, roll_id, use_roll_pool, raw_m=raw_m)
                final_p, grade = _apply_user_adjustments(eff_p, eff_g, adj, engine=engine, film=film)
                rp = coerce_render_params(final_p)
                crop_box = lr_crop if lr_crop is not None else (_auto_border_crop_norm(pv) if auto_crop else None)
                base_rgb = pv.base_info["rgb_render"] if film == "bw" else pv.inversion["base"]
                d_lo = pv.inversion["d_lo"]
                d_hi = pv.inversion["d_hi"]
                gamma = pv.inversion["gamma"]
                sc_clamp = pv.inversion.get("spike_clamp")
                t18_inv = pv.t18_inversion() if isinstance(rp, T19_FAMILY) else None
                base_info = pv.base_info
                img_region = pv.image_region
                wp_source = pv.report.get("white_point")
                inv_dict = pv.inversion

            # Step 2: Single full-resolution decode -> CROP FIRST -> fused chunked float32 invert+render!
            is_scan = is_scanner_dng(src, allow_frozen=True)
            if is_scan:
                fr = load_scanner(src, allow_frozen=True)
                rgb_lin = fr.rgb.astype(np.float32, copy=False)
                del fr
            else:
                with open_rawpy(src) as r:
                    rgb_lin = _postprocess_linear(r)

            # Crop BEFORE full-res inversion/rendering when no straighten angle (saves 25-40% pixels & RAM!)
            cropped_early = False
            if crop_box is not None and abs(lr_angle) < 0.05:
                H, W = rgb_lin.shape[:2]
                t_c, l_c, b_c, r_c = crop_box
                y0 = max(0, min(H - 2, int(round(t_c * H))))
                x0 = max(0, min(W - 2, int(round(l_c * W))))
                y1 = max(y0 + 1, min(H, int(round(b_c * H))))
                x1 = max(x0 + 1, min(W, int(round(r_c * W))))
                rgb_lin = np.ascontiguousarray(rgb_lin[y0:y1, x0:x1])
                cropped_early = True

            disp = _fused_invert_and_render(
                rgb_lin, base_rgb=base_rgb, d_lo=d_lo, d_hi=d_hi, gamma=gamma,
                spike_clamp=sc_clamp, rp=rp, film=film, t18_inv=t18_inv,
            )
            del rgb_lin

            # Step 3: Fast full-precision 16-bit darkroom grading (chunked 6-thread 33^3 float32 LUT + 4x blur)
            if not grade.is_identity():
                disp = apply_darkroom_grade(disp, grade, film=film, apply_vignette_here=False, fast=False, inplace=True)

            if not cropped_early and (crop_box is not None or abs(lr_angle) >= 0.05):
                disp = apply_lr_crop_angle(disp, crop_box, lr_angle, fast_preview=False)
            if rot90:
                disp = np.ascontiguousarray(np.rot90(disp, rot90))
            if lr_flip_h:
                disp = np.ascontiguousarray(disp[:, ::-1])
            if grade.vignette:
                disp = apply_vignette(disp, grade.vignette, inplace=True)

            # Step 4: Fast lossless TIFF-16 / JPEG / PNG write
            if fmt == "tiff16" and encoding == "linear":
                # Chunked in-place float32 sRGB->linear to avoid 1.7 GB float64 peak RAM spike
                h_d = disp.shape[0]
                for r0 in range(0, h_d, CHUNK_ROWS):
                    r1 = min(h_d, r0 + CHUNK_ROWS)
                    disp[r0:r1] = srgb_eotf(disp[r0:r1]).astype(np.float32)
            out_img = disp
            h_o, w_o = out_img.shape[:2]
            if max_edge is not None and max_edge > 0 and max(h_o, w_o) > max_edge:
                sc_e = max_edge / float(max(h_o, w_o))
                shape_out = [max(1, int(round(h_o * sc_e))), max(1, int(round(w_o * sc_e)))]
            else:
                shape_out = [h_o, w_o]
            ext = ".tif" if fmt == "tiff16" else (".jpg" if fmt == "jpeg" else ".png")
            stem = free_stem_ext(out_dir, src.stem + suffix, ext, check_json=write_sidecar)
            out_path = out_dir / f"{stem}{ext}"
            eff_encoding = encoding if fmt == "tiff16" else "srgb"
            export_image(out_path, out_img, fmt=fmt, encoding=eff_encoding, quality=quality, max_edge=max_edge, fast_tiff=True)
            del disp, out_img
            gc.collect()

            exif_copied = _copy_exif_metadata(src, out_path) if copy_exif else False
            sidecar_path = None
            if write_sidecar:
                rp_dict = rp.to_dict() if isinstance(rp, (RenderParams, RenderParamsT19, RenderParamsT24)) else final_p
                geom_rec = Geometry(rot90=rot90, angle=lr_angle, crop=crop_box)
                sc = dict(
                    schema="filmcolor.sidecar/0.2",
                    input=dict(path=str(src), sha256=sha256(src)),
                    base=base_info,
                    image_region=dict(region=list(img_region)),
                    inversion=dict(params=inv_dict, white_pct=WHITE_PCT,
                                   **({"white_point": wp_source} if wp_source else {})),
                    render=dict(
                        params=rp_dict,
                        source=dict(engine=engine, preset=preset, plugin="filmcolor_bridge"),
                        **({"t18_inversion": t18_inv} if t18_inv else {}),
                        **({"darkroom_grade": grade.to_dict()} if not grade.is_identity() else {}),
                    ),
                    output=dict(
                        path=str(out_path), sha256=sha256(out_path), format=fmt,
                        encoding=encoding if fmt == "tiff16" else "srgb",
                        bit_depth=16 if fmt == "tiff16" else 8, shape=shape_out,
                        **({"geometry": geom_rec.to_dict()} if not geom_rec.is_identity() else {}),
                    ),
                )
                js = out_dir / f"{stem}.json"
                js.write_text(json.dumps(sc, indent=1, ensure_ascii=False), encoding="utf-8")
                sidecar_path = str(js)

            return dict(
                source=str(src), output_path=str(out_path), sidecar_path=sidecar_path,
                format=fmt, shape=shape_out, engine=engine, preset=preset,
                exif_copied=exif_copied, qc_status="pass",
            )


class _BridgeHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30
    state: PluginBridgeState
    server_ref: _BridgeHTTPServer

    def log_message(self, fmt, *args):
        pass

    def _is_origin_allowed(self) -> tuple[bool, str | None]:
        origin = self.headers.get("Origin")
        if not origin:
            return True, None
        if origin == "null":
            return True, "null"
        u = urlparse(origin)
        if u.scheme in ("http", "https") and u.hostname in ("127.0.0.1", "localhost", "::1"):
            return True, origin
        return False, None

    def _cors_headers(self):
        ok, allow = self._is_origin_allowed()
        if ok and allow:
            self.send_header("Access-Control-Allow-Origin", allow)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        ok, _ = self._is_origin_allowed()
        if not ok:
            return self._send_json(dict(error="forbidden origin"), 403)
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_json(self, obj: dict, code: int = 200):
        safe_obj = _sanitize_json_obj(obj)
        raw = json.dumps(safe_obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self.send_response(code)
            self._cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass

    def _read_body(self) -> dict:
        raw_cl = self.headers.get("Content-Length")
        try:
            n = int(raw_cl) if raw_cl not in (None, "") else 0
        except ValueError as exc:
            raise ValueError(f"invalid Content-Length: {raw_cl!r}") from exc
        if n < 0 or n > 16 * 1024 * 1024:
            raise ValueError(f"invalid Content-Length: {n}")
        payload = self.rfile.read(n) if n > 0 else b""
        data = json.loads(payload or b"{}")
        if not isinstance(data, dict):
            raise ValueError("JSON request body must be an object")
        return data

    def do_GET(self):
        ok, _ = self._is_origin_allowed()
        if not ok:
            return self._send_json(dict(error="forbidden origin"), 403)
        u = urlparse(self.path)
        self.state.touch()
        try:
            if u.path == "/health":
                return self._send_json(dict(
                    ok=True, version=__version__, algorithm="T25", default_engine="t25", pid=os.getpid(),
                    engines=list(ENGINES),
                    presets=[dict(id=k, label=v["label"], desc=v["desc"]) for k, v in T24_STYLE_PRESETS.items()],
                ))
            self._send_json(dict(error="not found"), 404)
        except Exception as e:  # noqa: BLE001
            self._send_json(dict(error=f"{type(e).__name__}: {e}"), 400)

    def do_POST(self):
        ok, _ = self._is_origin_allowed()
        if not ok:
            return self._send_json(dict(error="forbidden origin"), 403)
        u = urlparse(self.path)
        self.state.touch()
        try:
            body = self._read_body()
            if u.path == "/shutdown":
                self._send_json(dict(ok=True, status="shutting_down"))
                threading.Thread(target=self.server_ref.shutdown, daemon=True).start()
                return
            if u.path == "/plugin/preview":
                with self.state.track_request():
                    return self._send_json(self.state.preview(body))
            if u.path == "/plugin/roll_analyze":
                with self.state.track_request():
                    return self._send_json(self.state.roll_analyze(body))
            if u.path == "/plugin/convert":
                with self.state.track_request():
                    return self._send_json(self.state.convert_one(body))
            self._send_json(dict(error="not found"), 404)
        except FileNotFoundError as e:
            self._send_json(dict(error=f"{type(e).__name__}: {e}"), 404)
        except Exception as e:  # noqa: BLE001
            self._send_json(dict(error=f"{type(e).__name__}: {e}"), 400)


def serve_bridge(port: int = DEFAULT_PORT, *, idle_timeout: int = DEFAULT_IDLE_TIMEOUT) -> _BridgeHTTPServer:
    st = PluginBridgeState()

    class BoundHandler(BridgeHandler):
        state = st
        server_ref = None

    srv = _BridgeHTTPServer(("127.0.0.1", port), BoundHandler)
    srv.state = st
    BoundHandler.server_ref = srv

    if idle_timeout and idle_timeout > 0:
        def watchdog():
            trimmed = False
            while True:
                time.sleep(min(30, max(1, idle_timeout // 4)))
                if st.active_requests > 0:
                    trimmed = False
                    continue
                idle = time.monotonic() - st.last_activity
                if idle > idle_timeout:
                    st.trim_caches(keep_displayed=False)
                    srv.shutdown()
                    break
                if idle > 300 and not trimmed:
                    st.trim_caches(keep_displayed=True)
                    trimmed = True
                elif idle < 60:
                    trimmed = False
        threading.Thread(target=watchdog, daemon=True).start()

    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description="FilmColor local bridge for Lightroom Classic & Photoshop plugins")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"loopback port (default {DEFAULT_PORT})")
    ap.add_argument("--idle-timeout", type=int, default=DEFAULT_IDLE_TIMEOUT,
                    help="auto-shutdown after N idle seconds (0 disables)")
    a = ap.parse_args(argv)
    srv = serve_bridge(a.port, idle_timeout=a.idle_timeout)
    actual_port = srv.server_address[1]
    info_file = _cache_root() / "plugin_bridge" / "bridge.json"
    info_file.parent.mkdir(parents=True, exist_ok=True)
    info_file.write_text(json.dumps(dict(pid=os.getpid(), port=actual_port, version=__version__)), encoding="utf-8")
    def _handle_sigterm(_signum, _frame):
        threading.Thread(target=srv.shutdown, daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except (ValueError, OSError):
        pass
    print(f"FilmColor T25 Plugin Bridge listening on http://127.0.0.1:{actual_port}/ (pid {os.getpid()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(srv, "state"):
            srv.state.wait_idle(timeout=30.0)
        _EXIFTOOL_DAEMON.close()
        srv.server_close()
        try:
            if info_file.exists() and json.loads(info_file.read_text()).get("pid") == os.getpid():
                info_file.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
