"""Darkroom post-grading, geometry (crop/straighten/flip), style presets, and multi-format export.

Pure NumPy + SciPy + Pillow. All operations are strict no-ops when at default values so existing
pinned RenderParams / RenderParamsT24 outputs remain bit-identical unless post-grading is active.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates, uniform_filter

from .export import DESCRIPTION, icc_profile, write_tiff16
from .render import (
    LUMA_F32,
    RenderParams,
    _apply_skin_protect,
    _apply_tone_curve,
    _apply_zonal_tones,
    _smoothstep,
    _validate_curve,
    eval_monotone_spline,
)

HSL_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
# Band hue centres in degrees [0, 360)
_HSL_CENTERS = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 235.0, 280.0, 325.0], dtype=np.float32)

# Precomputed 4096-point 1D grid for fused zonal + curve LUTs (accurate to < 1e-4)
_LUT1D_N = 4096
_LUT1D_X = np.linspace(0.0, 1.0, _LUT1D_N, dtype=np.float32)

# Precomputed 720-entry hue LUT grid (0.5 deg resolution)
_HUE_LUT_N = 720
_HUE_LUT_DEG = np.linspace(0.0, 360.0, _HUE_LUT_N, endpoint=False, dtype=np.float32)

# Shared bounded worker pool for chunked full-res ops (avoids per-call thread spawn overhead)
def _get_pool() -> ThreadPoolExecutor:
    from .invert import get_pixel_pool

    return get_pixel_pool()


def _run_row_chunks(h: int, fn, *, step: int = 128) -> None:
    ranges = [(r0, min(h, r0 + step)) for r0 in range(0, h, step)]
    if len(ranges) > 2:
        list(_get_pool().map(fn, ranges))
    else:
        for rr in ranges:
            fn(rr)

# Curated adaptive darkroom presets layered cleanly on top of locked T24 / T23 physical auto estimates.
# Static dicts serve as metadata + fallback when scene_metrics is omitted; resolve_adaptive_preset()
# modulates exposure centering, contrast, tone curve, dehaze, highlights, shadows, blacks, HSL density,
# split-tone, saturation/vibrance, tint cleanup, and skin protection smoothly per image.
T24_STYLE_PRESETS = {
    "frontier": dict(
        label="富士 Frontier 鲜润",
        desc="自适应 SP-3000 鲜润通透影调：翡翠通透绿、蜜桃清亮肤色、明澈天蓝与弹性反差（日常直出推荐）",
        camera_matrix_mix=0.45,
        offsets=dict(sat=0.05, contrast=0.06, shoulder=0.02, toe=-0.04, tint=0.008, temp=0.004),
        grade=dict(
            vibrance=0.14, highlights=-0.08, shadows=0.04, blacks=-0.02, whites=0.03, clarity=0.05,
            skin_protect=0.22,
            tone_curve={"rgb": [[0.0, 0.0], [0.24, 0.222], [0.50, 0.504], [0.76, 0.784], [1.0, 1.0]]},
            hsl={
                "red": [0.06, 0.14, -0.06],
                "orange": [0.03, 0.09, 0.14],
                "yellow": [-0.14, 0.05, 0.10],
                "green": [0.14, 0.18, 0.14],
                "aqua": [-0.08, 0.12, 0.05],
                "blue": [-0.14, 0.16, -0.10],
            },
            cg_shadows=[192.0, 0.10, -0.01],
            cg_highlights=[46.0, 0.12, 0.015],
            cg_blending=0.68,
        ),
    ),
    "leica": dict(
        label="德味徕卡 Leica Look",
        desc="自适应德味光学油润影调：深沉凝重红蓝、克制松烟橄榄绿、立体微反差与冷岩暗部/暖象牙高光",
        camera_matrix_mix=0.46,
        offsets=dict(ev=-0.05, contrast=0.09, sat=0.03, shoulder=0.07, toe=-0.07, tint=0.006),
        grade=dict(
            vibrance=0.10, highlights=-0.11, shadows=0.04, blacks=-0.04,
            clarity=0.14, dehaze=0.03, sharpen=0.12, vignette=-0.14, skin_protect=0.28,
            tone_curve={"rgb": [[0.0, 0.0], [0.22, 0.188], [0.50, 0.488], [0.78, 0.810], [1.0, 1.0]]},
            hsl={
                "red": [-0.14, 0.24, -0.24],
                "orange": [-0.03, 0.04, 0.06],
                "yellow": [-0.18, -0.18, -0.04],
                "green": [0.14, -0.24, -0.14],
                "aqua": [0.04, -0.18, -0.10],
                "blue": [-0.16, 0.22, -0.24],
            },
            cg_shadows=[218.0, 0.20, -0.025],
            cg_midtones=[36.0, 0.05, 0.0],
            cg_highlights=[42.0, 0.17, 0.015],
            cg_blending=0.66,
        ),
    ),
    "noritsu": dict(
        label="诺日士 Noritsu 通透",
        desc="自适应 HS-1800 空气感店扫：柔和丝滑高光、通透开放暗部、干净象牙白与克制高级色彩",
        camera_matrix_mix=0.42,
        offsets=dict(ev=0.03, sat=0.03, contrast=0.01, shoulder=0.10, toe=0.04, tint=0.006, temp=0.004),
        grade=dict(
            vibrance=0.10, highlights=-0.12, shadows=0.12, blacks=0.0, whites=0.02, skin_protect=0.28,
            hsl={
                "orange": [0.04, 0.05, 0.14],
                "yellow": [-0.12, -0.12, 0.12],
                "green": [0.06, -0.04, 0.10],
                "blue": [-0.10, 0.10, 0.02],
            },
            cg_shadows=[208.0, 0.06, 0.0],
            cg_highlights=[46.0, 0.13, 0.015],
            cg_blending=0.74,
        ),
    ),
    "linear": dict(
        label="线性 Linear",
        desc="自适应高宽容平坦底稿：最大化保留高光与暗部动态范围、中性色度且无风格化染调（专为 Lr/PS 深度自调设计）",
        camera_matrix_mix=0.40,
        offsets=dict(contrast=-0.20, sat=-0.08, shoulder=0.20, toe=0.08, tint=0.008),
        grade=dict(highlights=-0.18, shadows=0.16, blacks=0.04),
    ),
    "standard": dict(
        label="标准 Standard",
        desc="FilmColor T25 锁定物理基准，无额外风格偏移（最忠实原始反相结果）",
        offsets=dict(),
        grade=dict(),
    ),
    "natural": dict(
        label="自然 Natural",
        desc="自适应温润胶片本色：柔和光比、舒展暗部与阳光暖白高光，耐看无侵略性",
        offsets=dict(ev=0.02, contrast=-0.04, sat=0.0, shoulder=0.10),
        grade=dict(
            shadows=0.10, highlights=-0.10, vibrance=0.08, skin_protect=0.20,
            hsl={"yellow": [-0.10, -0.06, 0.08], "orange": [0.02, 0.05, 0.08]},
            cg_highlights=[40.0, 0.15, 0.015], cg_blending=0.72,
        ),
    ),
    "portrait": dict(
        label="人像 Portrait",
        desc="自适应 Portra 蜜桃通透肤色：润泽粉桃肌理、柔化绿植反光、暖金柔光高比与强肤色保护",
        offsets=dict(ev=0.05, contrast=-0.07, sat=-0.02, shoulder=0.13, toe=0.06, temp=0.008, tint=0.008),
        grade=dict(
            shadows=0.14, highlights=-0.14, skin_protect=0.48, vibrance=0.07,
            hsl={
                "red": [0.10, -0.06, 0.08],
                "orange": [0.06, 0.02, 0.20],
                "yellow": [-0.20, -0.22, 0.12],
                "green": [0.10, -0.24, 0.04],
                "blue": [-0.08, -0.06, 0.08],
            },
            cg_shadows=[212.0, 0.07, 0.0],
            cg_highlights=[38.0, 0.17, 0.015],
            cg_blending=0.72,
        ),
    ),
    "vivid": dict(
        label="鲜明 Vivid",
        desc="自适应反转片 Velvia/Ektar 浓郁毒色：宝石深邃蓝红绿、晶莹通透微反差，高饱和自动防溢色",
        camera_matrix_mix=0.47,
        offsets=dict(ev=-0.02, contrast=0.11, sat=0.07, shoulder=0.02, toe=-0.06),
        grade=dict(
            vibrance=0.18, highlights=-0.10, blacks=-0.04, clarity=0.10, skin_protect=0.22,
            tone_curve={"rgb": [[0.0, 0.0], [0.23, 0.200], [0.50, 0.498], [0.77, 0.804], [1.0, 1.0]]},
            hsl={
                "red": [-0.08, 0.26, -0.18],
                "orange": [-0.06, 0.18, -0.03],
                "yellow": [-0.10, 0.10, 0.02],
                "green": [0.12, 0.22, -0.10],
                "aqua": [-0.06, 0.20, -0.10],
                "blue": [-0.14, 0.28, -0.28],
                "magenta": [-0.04, 0.14, -0.08],
            },
            cg_shadows=[214.0, 0.10, -0.015],
            cg_highlights=[44.0, 0.10, 0.01],
            cg_blending=0.68,
        ),
    ),
    "print_classic": dict(
        label="经典印相 Print",
        desc="自适应光学暗房 RA-4 印相：雕塑感经典 S 曲线、扎实丝绒黑位、巴莱塔温润纸白与细腻微锐",
        offsets=dict(ev=-0.02, contrast=0.09, sat=0.02, shoulder=0.02, toe=-0.05),
        grade=dict(
            blacks=-0.06, whites=0.03, highlights=-0.09, clarity=0.07, sharpen=0.12, skin_protect=0.18,
            tone_curve={"rgb": [[0.0, 0.0], [0.24, 0.202], [0.50, 0.500], [0.76, 0.800], [1.0, 1.0]]},
            hsl={
                "yellow": [-0.10, -0.10, 0.02],
                "green": [0.08, -0.12, -0.08],
                "blue": [-0.08, 0.08, -0.14],
            },
            cg_shadows=[224.0, 0.08, -0.015],
            cg_highlights=[44.0, 0.18, 0.015],
            cg_blending=0.70,
        ),
    ),
    "cinema": dict(
        label="电影质感 Cinema",
        desc="自适应 Vision3 2383 银幕色调：油青深邃暗部与钨丝暖金高光分离、胶片宽容肩部与有机颗粒",
        offsets=dict(ev=-0.02, contrast=0.04, sat=-0.01, shoulder=0.15, toe=0.03),
        grade=dict(
            blacks=0.04, highlights=-0.14, shadows=0.05, grain=0.08, clarity=0.06, skin_protect=0.30,
            hsl={
                "red": [0.06, 0.16, -0.14],
                "orange": [0.02, 0.06, 0.08],
                "yellow": [-0.22, -0.16, -0.04],
                "green": [0.18, -0.26, -0.18],
                "blue": [-0.22, 0.12, -0.14],
            },
            cg_shadows=[196.0, 0.30, -0.025],
            cg_midtones=[28.0, 0.05, 0.0],
            cg_highlights=[35.0, 0.26, 0.015],
            cg_balance=-0.08,
            cg_blending=0.62,
        ),
    ),
}


def analyze_scene_metrics(
    display: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    above_white: float | None = None,
) -> dict[str, float]:
    """Compute compact, deterministic tonal & chromatic scene statistics on an sRGB [0, 1] display preview (~3 ms)."""
    arr = np.asarray(display, dtype=np.float32)
    if arr.ndim == 2:
        h, w = arr.shape
        step = max(1, max(h, w) // 192)
        sub_1 = arr[::step, ::step]
        sub = np.repeat(sub_1[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        h, w = arr.shape[:2]
        step = max(1, max(h, w) // 192)
        sub = arr[::step, ::step, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        h, w = arr.shape[:2]
        step = max(1, max(h, w) // 192)
        sub = np.repeat(arr[::step, ::step], 3, axis=-1)
    else:
        raise ValueError(f"analyze_scene_metrics expects 2D or 3D image, got shape {arr.shape}")
    if mask is not None and mask.shape[:2] == (h, w):
        m = np.asarray(mask[::step, ::step], dtype=bool)
        px = sub[m] if int(m.sum()) >= 64 else sub.reshape(-1, 3)
    else:
        px = sub.reshape(-1, 3)
    finite_rows = np.all(np.isfinite(px), axis=1)
    if not np.all(finite_rows):
        px = px[finite_rows]
    if px.size > 0:
        px = np.clip(px, np.float32(0.0), np.float32(1.0))
    if px.size == 0:
        return dict(
            luma_p02=0.04, luma_p05=0.08, luma_p25=0.25, luma_p50=0.45, luma_p75=0.65,
            luma_p95=0.82, luma_p98=0.90, dr=0.74, mid_dr=0.40, hi_clip=0.0, sh_crush=0.02,
            chroma_p50=0.25, chroma_p90=0.45, skin_share=0.0, blue_share=0.0,
            green_share=0.0, warm_share=0.0, warm_bias=0.0, green_bias=0.0,
            shadow_warm_bias=0.0, hi_warm_bias=0.0,
        )

    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    y = r * LUMA_F32[0] + g * LUMA_F32[1] + b * LUMA_F32[2]
    p02, p05, p25, p50, p75, p95, p98 = (float(v) for v in np.percentile(y, [2, 5, 25, 50, 75, 95, 98]))

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    sat = delta / np.maximum(mx, np.float32(1e-4))
    c50, c90 = (float(v) for v in np.percentile(sat, [50, 90]))

    hi_clip = max(
        float(np.mean(mx >= np.float32(0.985))),
        float(above_white) if (above_white is not None and np.isfinite(above_white)) else 0.0,
    )
    sh_crush = float(np.mean(y <= np.float32(0.04)))

    mid = (y > np.float32(0.14)) & (y < np.float32(0.86))
    if int(mid.sum()) >= 32:
        rm, gm, bm = r[mid], g[mid], b[mid]
        warm_bias = float(np.mean(rm - bm))
        neut = mid & (sat < np.float32(0.24))
        if int(neut.sum()) >= 32:
            rn, gn, bn = r[neut], g[neut], b[neut]
            green_bias = float(np.mean(gn - np.float32(0.5) * (rn + bn)))
        else:
            green_bias = float(np.mean(gm - np.float32(0.5) * (rm + bm))) * 0.35
    else:
        warm_bias, green_bias = 0.0, 0.0

    # Zonal warm/cool cast in shadows & highlights so split-toning adapts rather than double-tinting
    sh_zone = (y >= np.float32(0.03)) & (y <= np.float32(0.28))
    shadow_warm_bias = float(np.mean(r[sh_zone] - b[sh_zone])) if int(sh_zone.sum()) >= 32 else warm_bias * 0.5
    hi_zone = (y >= np.float32(0.70)) & (y <= np.float32(0.96))
    hi_warm_bias = float(np.mean(r[hi_zone] - b[hi_zone])) if int(hi_zone.sum()) >= 32 else warm_bias * 0.5

    # Natural skin-tone window: R > G > B, hue in [6, 42] deg, moderate saturation & luma
    skin_cand = (
        (r > g) & (g > b)
        & (sat >= np.float32(0.12)) & (sat <= np.float32(0.62))
        & (y >= np.float32(0.18)) & (y <= np.float32(0.85))
    )
    if np.any(skin_cand):
        d_s = np.maximum(delta[skin_cand], np.float32(1e-5))
        hue_s = np.float32(60.0) * ((g[skin_cand] - b[skin_cand]) / d_s)
        skin_share = float(np.sum((hue_s >= np.float32(6.0)) & (hue_s <= np.float32(42.0)))) / float(len(y))
    else:
        skin_share = 0.0

    # Dominant chromatic families for adaptive hue sculpting
    blue_share = float(np.mean((b > g) & (b > r) & (sat >= np.float32(0.14)) & (y >= np.float32(0.20))))
    green_share = float(np.mean((g > r) & (g > b) & (sat >= np.float32(0.14)) & (y >= np.float32(0.08))))
    warm_share = float(np.mean((r > b) & (r >= g) & (sat >= np.float32(0.18)) & (y >= np.float32(0.10))))

    return dict(
        luma_p02=p02, luma_p05=p05, luma_p25=p25, luma_p50=p50, luma_p75=p75,
        luma_p95=p95, luma_p98=p98, dr=max(0.0, p95 - p05), mid_dr=max(0.0, p75 - p25),
        hi_clip=hi_clip, sh_crush=sh_crush, chroma_p50=c50, chroma_p90=c90,
        skin_share=skin_share, blue_share=blue_share, green_share=green_share, warm_share=warm_share,
        warm_bias=warm_bias, green_bias=green_bias,
        shadow_warm_bias=shadow_warm_bias, hi_warm_bias=hi_warm_bias,
    )


def resolve_adaptive_preset(
    preset: str,
    film: str = "color",
    scene_metrics: dict[str, float] | None = None,
) -> dict:
    """T25-calibrated adaptive preset resolver: modulates color density, hue separation, micro-contrast,
    split-toning, and tonal roll-off continuously per image (falls back cleanly to T24_STYLE_PRESETS when None)."""
    spec = T24_STYLE_PRESETS.get(preset) or T24_STYLE_PRESETS["standard"]
    if preset == "standard" or scene_metrics is None:
        mix = spec.get("camera_matrix_mix")
        offsets = dict(spec["offsets"])
        grade = copy.deepcopy(spec["grade"])
        if film == "bw":
            mix = None
            for ck in ("vibrance", "skin_protect", "hsl", "cg_shadows", "cg_midtones", "cg_highlights", "cg_balance", "cg_blending"):
                grade.pop(ck, None)
        return dict(camera_matrix_mix=mix, offsets=offsets, grade=grade)

    m = scene_metrics
    dr = float(m.get("dr", 0.72))
    mid_dr = float(m.get("mid_dr", 0.36))
    p02 = float(m.get("luma_p02", 0.04))
    p05 = float(m.get("luma_p05", 0.08))
    p50 = float(m.get("luma_p50", 0.45))
    p98 = float(m.get("luma_p98", 0.90))
    hi_clip = float(m.get("hi_clip", 0.0))
    sh_crush = float(m.get("sh_crush", 0.0))
    c50 = float(m.get("chroma_p50", 0.22))
    c90 = float(m.get("chroma_p90", 0.48))
    skin_share = float(m.get("skin_share", 0.0))
    blue_share = float(m.get("blue_share", 0.0))
    green_share = float(m.get("green_share", 0.0))
    warm_share = float(m.get("warm_share", 0.0))
    warm_bias = float(m.get("warm_bias", 0.0))
    green_bias = float(m.get("green_bias", 0.0))
    sh_warm = float(m.get("shadow_warm_bias", warm_bias * 0.5))
    hi_warm = float(m.get("hi_warm_bias", warm_bias * 0.5))

    # --- T25-calibrated continuous scene drivers ---
    # Combine global DR flatness and midtone flatness so soft overcast scenes get crisp separation
    # while punchy sunlit scenes stay controlled and never brittle.
    flat_global = float(np.clip((0.72 - dr) / 0.28, -0.60, 1.0))
    flat_mid = float(np.clip((0.36 - mid_dr) / 0.22, -0.55, 1.0))
    flatness = float(np.clip(0.65 * flat_global + 0.35 * flat_mid, -0.60, 1.0))

    hi_pressure = float(np.clip((p98 - 0.87) / 0.12, 0.0, 1.0) * 0.55 + np.clip(hi_clip / 0.12, 0.0, 1.0) * 0.45)
    sh_lift_need = float(np.clip(max((0.10 - p05) / 0.08, sh_crush / 0.14), 0.0, 1.0))
    low_key = float(np.clip((0.36 - p50) / 0.18, 0.0, 1.0))
    high_key = float(np.clip((p50 - 0.62) / 0.16, 0.0, 1.0))
    high_key_guard = float(np.clip((0.64 - p50) / 0.14, 0.0, 1.0))
    narrow_dr_guard = float(np.clip((dr - 0.42) / 0.14, 0.0, 1.0))

    milky_black = float(
        np.clip((p05 - 0.11) / 0.12, 0.0, 1.0)
        * np.clip((p50 - 0.30) / 0.16, 0.20, 1.0)
        * high_key_guard
        * narrow_dr_guard
    )
    veil_need = float(np.clip((min(p02, p05) - 0.07) / 0.14, 0.0, 1.0) * high_key_guard * narrow_dr_guard)

    # T25 native chroma is richer (~0.885 vs old T24 0.806): decouple saturation ceiling from hue/density craft!
    chroma_headroom = float(np.clip((0.52 - c90) / 0.30, -0.85, 1.0))
    mid_chroma_room = float(np.clip((0.26 - c50) / 0.18, -0.60, 1.0))
    chroma_guard = float(np.clip((c90 - 0.60) / 0.30, 0.0, 1.0))
    # Saturation pushes back smoothly when c90 is high, while hue/luminance signature stays 80-100% alive!
    sat_sc = float(np.clip(1.0 - 0.62 * chroma_guard + 0.18 * max(chroma_headroom, 0.0), 0.35, 1.25))
    hue_sc = float(np.clip(1.0 - 0.18 * chroma_guard, 0.80, 1.05))

    skin_present = float(np.clip((skin_share - 0.03) / 0.15, 0.0, 1.0))
    blue_present = float(np.clip((blue_share - 0.04) / 0.22, 0.0, 1.0))
    green_present = float(np.clip((green_share - 0.04) / 0.22, 0.0, 1.0))
    warm_present = float(np.clip((warm_share - 0.05) / 0.24, 0.0, 1.0))

    green_cast = float(np.clip(green_bias / 0.032, 0.0, 1.0))
    magenta_cast = float(np.clip(-green_bias / 0.032, 0.0, 1.0))
    cool_need = float(np.clip(-warm_bias / 0.10, -0.5, 1.0))
    warm_excess = float(np.clip((warm_bias - 0.07) / 0.14, 0.0, 1.0))
    dark_mid = float(np.clip((0.44 - p50) / 0.20, -0.5, 1.0))

    # Zonal split-tone headroom: back off cool shadow tint if shadows are already cold;
    # back off warm highlight glow if highlights are already warm golden!
    sh_cool_room = float(np.clip(1.0 + sh_warm / 0.08, 0.35, 1.25))
    hi_warm_room = float(np.clip(1.0 - hi_warm / 0.09, 0.38, 1.25))
    # Strong toning attenuation on ultra-narrow pale empty-sky frames (e.g. A025) so skies stay pristine
    sky_blank_guard = float(np.clip((c90 - 0.12) / 0.16, 0.0, 1.0))
    tone_body = float(np.clip(0.14 + 0.86 * narrow_dr_guard * (0.35 + 0.65 * sky_blank_guard), 0.14, 1.0))

    mix: float | None = None
    offsets: dict[str, float] = {}
    grade: dict = {}

    if preset == "leica":
        # Authentic Leica M / APO-Summicron film aesthetic:
        # Dense, painterly midtones (never washed-out), oil-rich garnet reds & Prussian blues,
        # aristocratic pine-olive greens, sculpted micro-contrast, slate-blue shadows vs alabaster highlights.
        mix = round(0.42 + 0.04 * (1.0 - chroma_guard) + 0.04 * max(chroma_headroom, 0.0), 3)
        # Keep rich midtone weight on normal scenes; gently protect only very dark low-key scenes
        ev_adj = round((-0.055 * (1.0 - 0.75 * low_key) * (1.0 - 0.45 * sh_lift_need) + 0.055 * low_key * (1.0 - 0.55 * hi_pressure) - 0.03 * high_key) * (0.35 + 0.65 * narrow_dr_guard), 4)
        s_low = round(0.22 - (0.030 + 0.018 * max(flatness, -0.35)) * (1.0 - 0.72 * sh_lift_need) * tone_body, 4)
        s_hi = round(0.78 + (0.028 + 0.014 * max(flatness, -0.35)) * (1.0 - 0.55 * hi_pressure) * tone_body, 4)
        offsets = dict(
            ev=ev_adj,
            contrast=round(0.08 + 0.09 * flatness - 0.03 * sh_lift_need, 4),
            sat=round(0.025 + 0.07 * chroma_headroom - 0.05 * chroma_guard, 4),
            shoulder=round(0.07 + 0.11 * hi_pressure, 4),
            toe=round(-0.07 - 0.09 * milky_black + 0.18 * sh_lift_need, 4),
            tint=round(0.002 + 0.026 * green_cast - 0.042 * magenta_cast, 4),
            temp=round(0.002 + 0.008 * max(cool_need, 0.0) - 0.015 * warm_excess, 4),
        )
        hsl = {
            "red": [round(-0.14 * hue_sc, 3), round((0.22 + 0.08 * max(chroma_headroom, 0.0)) * sat_sc, 3), round((-0.22 - 0.06 * warm_present) * hue_sc, 3)],
            "orange": [round(-0.03 * hue_sc, 3), round((0.03 + 0.05 * max(chroma_headroom, 0.0)) * sat_sc, 3), round(0.06 + 0.05 * skin_present, 3)],
            "yellow": [round(-0.18 * hue_sc, 3), round(-0.18 - 0.05 * warm_present, 3), -0.04],
            "green": [round(0.14 * hue_sc, 3), round(-0.22 - 0.08 * green_present + 0.06 * max(chroma_headroom, 0.0), 3), round(-0.14 + 0.08 * low_key, 3)],
            "aqua": [0.04, -0.18, -0.10],
            "blue": [round(-0.16 * hue_sc, 3), round((0.20 + 0.08 * max(chroma_headroom, 0.0)) * sat_sc, 3), round((-0.22 - 0.10 * blue_present - 0.04 * hi_pressure) * hue_sc, 3)],
        }
        grade = dict(
            vibrance=round(0.10 + 0.10 * chroma_headroom + 0.04 * max(mid_chroma_room, 0.0) - 0.07 * chroma_guard, 4),
            highlights=round(-0.10 - 0.19 * hi_pressure, 4),
            shadows=round(0.03 + 0.18 * sh_lift_need + 0.05 * low_key, 4),
            blacks=round(-0.04 - 0.06 * milky_black + 0.10 * sh_lift_need, 4),
            clarity=round(0.12 + 0.08 * max(flatness, 0.0) * narrow_dr_guard, 4),
            dehaze=round(0.02 * narrow_dr_guard + 0.07 * veil_need, 4),
            sharpen=0.12,
            vignette=round(-0.14 * (1.0 - 0.50 * low_key) * narrow_dr_guard, 3),
            skin_protect=round(0.24 + 0.28 * skin_present, 4),
            tone_curve={"rgb": [[0.0, 0.0], [0.22, s_low], [0.50, round(0.488 + 0.008 * low_key, 4)], [0.78, s_hi], [1.0, 1.0]]},
            hsl=hsl,
            cg_shadows=[218.0, round(0.20 * sh_cool_room * tone_body, 3), round(-0.025 * (1.0 - 0.80 * sh_lift_need) * tone_body, 4)],
            cg_midtones=[36.0, round(0.05 * hi_warm_room * tone_body, 3), 0.0],
            cg_highlights=[42.0, round(0.17 * hi_warm_room * tone_body, 3), round(0.015 * (1.0 - 0.50 * hi_pressure), 4)],
            cg_blending=0.66,
        )
    elif preset == "linear":
        # Wide-latitude, flat, neutral 16-bit grading canvas: zero crushed blacks, zero clipped highlights,
        # zero stylized HSL/split-tone/vignette/grain/sharpen so colorists have maximum headroom in Lr/PS.
        mix = 0.40
        offsets = dict(
            ev=round(0.04 * (1.0 - 0.65 * hi_pressure) + 0.06 * max(dark_mid, 0.0) * (1.0 - 0.65 * hi_pressure) - 0.05 * high_key, 4),
            contrast=round((-0.20 - 0.06 * max(-flatness, 0.0)) * float(np.clip((dr - 0.15) / 0.35, 0.30, 1.0)), 4),
            sat=round(-0.08 + 0.04 * max(chroma_headroom, 0.0) - 0.03 * chroma_guard, 4),
            shoulder=round(0.18 + 0.12 * hi_pressure, 4),
            toe=round(0.08 + 0.12 * sh_lift_need - 0.06 * milky_black, 4),
            tint=round(0.020 * green_cast - 0.018 * magenta_cast, 4),
        )
        grade = dict(
            highlights=round(-0.16 - 0.18 * hi_pressure, 4),
            shadows=round(0.14 + 0.14 * sh_lift_need, 4),
            blacks=round(0.04 * (1.0 - milky_black) - 0.02 * milky_black, 4),
        )
    elif preset == "frontier":
        # Iconic Fuji Frontier SP-3000 aesthetic:
        # Crystalline luminous jade-emerald greens, translucent creamy-peach skin, airy cerulean skies,
        # springy elastic S-curve, clean neutral-ivory whites.
        mix = round(0.41 + 0.04 * (1.0 - chroma_guard) + 0.04 * max(chroma_headroom, 0.0), 3)
        ev_adj = round(0.015 * (1.0 - 0.70 * hi_pressure) + 0.05 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.04 * high_key, 4)
        s_low = round(0.24 - (0.018 + 0.012 * max(flatness, -0.35)) * (1.0 - 0.72 * sh_lift_need) * tone_body, 4)
        s_hi = round(0.76 + (0.022 + 0.012 * max(flatness, -0.35)) * (1.0 - 0.55 * hi_pressure) * tone_body, 4)
        offsets = dict(
            ev=ev_adj,
            contrast=round(0.055 + 0.085 * flatness - 0.025 * sh_lift_need, 4),
            sat=round(0.045 + 0.08 * chroma_headroom - 0.06 * chroma_guard, 4),
            shoulder=round(0.02 + 0.12 * hi_pressure, 4),
            toe=round(-0.04 - 0.07 * milky_black + 0.16 * sh_lift_need, 4),
            tint=round(0.004 + 0.026 * green_cast - 0.042 * magenta_cast, 4),
            temp=round(0.002 + 0.008 * max(cool_need, 0.0) - 0.014 * warm_excess, 4),
        )
        hsl = {
            "red": [round(0.06 * hue_sc, 3), round((0.14 + 0.06 * max(chroma_headroom, 0.0)) * sat_sc, 3), -0.06],
            "orange": [round(0.03 * hue_sc, 3), round((0.08 + 0.05 * max(chroma_headroom, 0.0)) * sat_sc, 3), round(0.12 + 0.08 * skin_present, 3)],
            "yellow": [round(-0.14 * hue_sc, 3), round((0.05 + 0.04 * max(chroma_headroom, 0.0)) * sat_sc, 3), 0.10],
            "green": [round(0.14 * hue_sc, 3), round((0.18 + 0.06 * max(chroma_headroom, 0.0)) * sat_sc, 3), round(0.14 + 0.06 * green_present, 3)],
            "aqua": [round(-0.08 * hue_sc, 3), round(0.12 * sat_sc, 3), 0.05],
            "blue": [round(-0.14 * hue_sc, 3), round((0.15 + 0.07 * max(chroma_headroom, 0.0)) * sat_sc, 3), round((-0.09 - 0.08 * blue_present - 0.04 * hi_pressure) * hue_sc, 3)],
        }
        grade = dict(
            vibrance=round(0.13 + 0.10 * chroma_headroom + 0.04 * max(mid_chroma_room, 0.0) - 0.08 * chroma_guard, 4),
            highlights=round(-0.07 - 0.19 * hi_pressure, 4),
            shadows=round(0.04 + 0.18 * sh_lift_need, 4),
            blacks=round(-0.02 - 0.05 * milky_black + 0.09 * sh_lift_need, 4),
            whites=round(0.035 * (1.0 - hi_pressure), 4),
            clarity=round(0.05 + 0.05 * max(flatness, 0.0) * narrow_dr_guard, 4),
            dehaze=round(0.06 * veil_need, 4),
            skin_protect=round(0.20 + 0.24 * skin_present, 4),
            tone_curve={"rgb": [[0.0, 0.0], [0.24, s_low], [0.50, 0.504], [0.76, s_hi], [1.0, 1.0]]},
            hsl=hsl,
            cg_shadows=[192.0, round(0.10 * sh_cool_room * tone_body, 3), round(-0.01 * (1.0 - 0.80 * sh_lift_need), 4)],
            cg_highlights=[46.0, round(0.12 * hi_warm_room * tone_body, 3), round(0.015 * (1.0 - 0.50 * hi_pressure), 4)],
            cg_blending=0.68,
        )
    elif preset == "noritsu":
        # Classic Noritsu HS-1800 airy lab-scan look:
        # Silky highlight roll-off, open breathable shadows, clean neutral-ivory whites, refined pastel clarity.
        mix = round(0.39 + 0.03 * (1.0 - chroma_guard) + 0.03 * max(chroma_headroom, 0.0), 3)
        offsets = dict(
            ev=round(0.03 * (1.0 - 0.65 * hi_pressure) + 0.06 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.04 * high_key, 4),
            contrast=round(0.01 + 0.065 * flatness - 0.02 * sh_lift_need, 4),
            sat=round(0.03 + 0.06 * chroma_headroom - 0.05 * chroma_guard, 4),
            shoulder=round(0.10 + 0.12 * hi_pressure, 4),
            toe=round(0.04 - 0.08 * milky_black + 0.12 * sh_lift_need, 4),
            tint=round(0.002 + 0.024 * green_cast - 0.024 * magenta_cast, 4),
            temp=round(0.002 + 0.007 * max(cool_need, 0.0) - 0.013 * warm_excess, 4),
        )
        grade = dict(
            vibrance=round(0.10 + 0.08 * chroma_headroom - 0.06 * chroma_guard, 4),
            highlights=round(-0.11 - 0.20 * hi_pressure, 4),
            shadows=round(0.11 + 0.16 * sh_lift_need, 4),
            blacks=round(0.01 * (1.0 - milky_black) - 0.06 * milky_black + 0.05 * sh_lift_need, 4),
            whites=round(0.02 * (1.0 - hi_pressure), 4),
            skin_protect=round(0.24 + 0.26 * skin_present, 4),
            hsl={
                "orange": [round(0.04 * hue_sc, 3), round(0.05 * sat_sc, 3), round(0.12 + 0.08 * skin_present, 3)],
                "yellow": [round(-0.12 * hue_sc, 3), -0.12, 0.12],
                "green": [round(0.06 * hue_sc, 3), round(-0.04 + 0.06 * max(chroma_headroom, 0.0), 3), 0.10],
                "blue": [round(-0.10 * hue_sc, 3), round((0.09 + 0.05 * max(chroma_headroom, 0.0)) * sat_sc, 3), round(0.03 - 0.09 * blue_present, 3)],
            },
            cg_shadows=[206.0, round(0.11 * sh_cool_room * tone_body, 3), 0.008],
            cg_highlights=[48.0, round(0.08 * hi_warm_room * tone_body, 3), round(0.020 * (1.0 - 0.50 * hi_pressure), 4)],
            cg_blending=0.74,
        )
    elif preset == "natural":
        # Sun-kissed, unforced analog warmth: gentle tone, open shadows, subtle warm highlight paper glow.
        offsets = dict(
            ev=round(0.02 * (1.0 - 0.65 * hi_pressure) + 0.05 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.04 * high_key, 4),
            contrast=round(-0.04 + 0.06 * flatness, 4),
            temp=round(0.012 + 0.010 * max(cool_need, 0.0) - 0.016 * warm_excess, 4),
            sat=round(0.0 + 0.045 * max(chroma_headroom, 0.0) - 0.035 * chroma_guard, 4),
            shoulder=round(0.10 + 0.10 * hi_pressure, 4),
            tint=round(0.020 * green_cast - 0.018 * magenta_cast, 4),
        )
        grade = dict(
            shadows=round(0.09 + 0.14 * sh_lift_need, 4),
            highlights=round(-0.09 - 0.16 * hi_pressure, 4),
            blacks=round(-0.04 * milky_black + 0.04 * sh_lift_need, 4),
            vibrance=round(0.07 + 0.06 * max(chroma_headroom, 0.0) - 0.04 * chroma_guard, 4),
            skin_protect=round(0.18 + 0.18 * skin_present, 4),
            hsl={
                "yellow": [-0.10, -0.06, 0.08],
                "orange": [0.02, round(0.05 * sat_sc, 3), round(0.07 + 0.05 * skin_present, 3)],
            },
            cg_highlights=[39.0, round(0.20 * hi_warm_room * tone_body, 3), round(0.016 * (1.0 - 0.50 * hi_pressure), 4)],
            cg_blending=0.72,
        )
    elif preset == "portrait":
        # Kodak Portra 400 overexposed +1/3 stop aesthetic:
        # Luminous warm-peach porcelain skin, calm sage-olive foliage, creamy golden highlight roll-off.
        offsets = dict(
            ev=round(0.05 * (1.0 - 0.60 * hi_pressure) + 0.07 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.04 * high_key, 4),
            contrast=round(-0.07 + 0.05 * flatness, 4),
            sat=round(-0.02 + 0.045 * max(chroma_headroom, 0.0) - 0.04 * chroma_guard, 4),
            shoulder=round(0.13 + 0.10 * hi_pressure, 4),
            toe=round(0.06 - 0.06 * milky_black + 0.10 * sh_lift_need, 4),
            temp=round(0.006 + 0.009 * max(cool_need, 0.0) - 0.014 * warm_excess, 4),
            tint=round(0.005 + 0.024 * green_cast - 0.020 * magenta_cast, 4),
        )
        grade = dict(
            shadows=round(0.13 + 0.15 * sh_lift_need, 4),
            highlights=round(-0.13 - 0.18 * hi_pressure, 4),
            blacks=round(0.02 * (1.0 - milky_black) - 0.05 * milky_black + 0.04 * sh_lift_need, 4),
            skin_protect=round(0.38 + 0.30 * skin_present, 4),
            vibrance=round(0.07 + 0.06 * max(chroma_headroom, 0.0) - 0.04 * chroma_guard, 4),
            hsl={
                "red": [round(0.10 * hue_sc, 3), -0.06, 0.08],
                "orange": [round(0.06 * hue_sc, 3), round(-0.05 * chroma_guard + 0.04 * max(chroma_headroom, 0.0), 3), round(0.18 + 0.12 * skin_present, 3)],
                "yellow": [round(-0.20 * hue_sc, 3), -0.22, 0.12],
                "green": [round(0.10 * hue_sc, 3), round(-0.22 - 0.08 * green_present, 3), 0.04],
                "blue": [-0.08, -0.06, 0.08],
            },
            cg_shadows=[212.0, round(0.07 * sh_cool_room * tone_body, 3), 0.0],
            cg_highlights=[38.0, round(0.17 * hi_warm_room * tone_body, 3), round(0.015 * (1.0 - 0.50 * hi_pressure), 4)],
            cg_blending=0.72,
        )
    elif preset == "vivid":
        # Fujichrome Velvia 50 / Kodak Ektar 100 slide-film jewel density:
        # Deep sapphire skies, ruby reds, deep emerald greens (rich + dark luminance = jewel, never neon),
        # punchy elastic S-curve, strong chroma-clipping & skin-tone guard.
        mix = round(0.43 + 0.04 * (1.0 - chroma_guard) + 0.05 * max(chroma_headroom, 0.0), 3)
        s_low = round(0.23 - (0.028 + 0.016 * max(flatness, -0.35)) * (1.0 - 0.72 * sh_lift_need) * tone_body, 4)
        s_hi = round(0.77 + (0.030 + 0.014 * max(flatness, -0.35)) * (1.0 - 0.55 * hi_pressure) * tone_body, 4)
        offsets = dict(
            ev=round(-0.025 * (1.0 - 0.70 * low_key) + 0.04 * low_key * (1.0 - 0.55 * hi_pressure) - 0.04 * high_key, 4),
            contrast=round(0.10 + 0.10 * flatness - 0.03 * sh_lift_need, 4),
            sat=round(0.065 + 0.10 * chroma_headroom - 0.075 * chroma_guard, 4),
            shoulder=round(0.02 + 0.12 * hi_pressure, 4),
            toe=round(-0.06 - 0.10 * milky_black + 0.14 * sh_lift_need, 4),
            tint=round(0.002 + 0.024 * green_cast - 0.024 * magenta_cast, 4),
        )
        grade = dict(
            vibrance=round(0.17 + 0.12 * chroma_headroom + 0.04 * max(mid_chroma_room, 0.0) - 0.10 * chroma_guard, 4),
            highlights=round(-0.09 - 0.19 * hi_pressure, 4),
            shadows=round(0.02 + 0.14 * sh_lift_need, 4),
            blacks=round(-0.04 - 0.07 * milky_black + 0.09 * sh_lift_need, 4),
            clarity=round(0.09 + 0.06 * max(flatness, 0.0) * narrow_dr_guard, 4),
            dehaze=round(0.08 * veil_need, 4),
            skin_protect=round(0.22 + 0.28 * skin_present, 4),
            tone_curve={"rgb": [[0.0, 0.0], [0.23, s_low], [0.50, 0.498], [0.77, s_hi], [1.0, 1.0]]},
            hsl={
                "red": [round(-0.08 * hue_sc, 3), round((0.24 + 0.08 * max(chroma_headroom, 0.0)) * sat_sc, 3), round(-0.18 * hue_sc, 3)],
                "orange": [round(-0.06 * hue_sc, 3), round((0.16 + 0.06 * max(chroma_headroom, 0.0)) * sat_sc * (1.0 - 0.45 * skin_present), 3), round(-0.03 + 0.09 * skin_present, 3)],
                "yellow": [round(-0.10 * hue_sc, 3), round(0.10 * sat_sc, 3), 0.02],
                "green": [round(0.12 * hue_sc, 3), round((0.20 + 0.06 * max(chroma_headroom, 0.0)) * sat_sc, 3), round(-0.10 * hue_sc, 3)],
                "aqua": [round(-0.06 * hue_sc, 3), round(0.20 * sat_sc, 3), -0.10],
                "blue": [round(-0.14 * hue_sc, 3), round((0.26 + 0.08 * max(chroma_headroom, 0.0)) * sat_sc, 3), round((-0.26 - 0.12 * blue_present) * hue_sc, 3)],
                "magenta": [-0.04, round(0.14 * sat_sc, 3), -0.08],
            },
            cg_shadows=[214.0, round(0.10 * sh_cool_room * tone_body, 3), round(-0.015 * (1.0 - 0.80 * sh_lift_need), 4)],
            cg_highlights=[44.0, round(0.10 * hi_warm_room * tone_body, 3), round(0.01 * (1.0 - 0.50 * hi_pressure), 4)],
            cg_blending=0.68,
        )
    elif preset == "print_classic":
        # Optical RA-4 darkroom print on warm baryta paper:
        # Authoritative sculpted S-curve, velvety D-max blacks, warm paper ivory highlights, quiet color poise.
        s_amp = (0.034 + 0.018 * max(flatness, -0.35)) * tone_body
        offsets = dict(
            ev=round((-0.02 * (1.0 - 0.70 * low_key) * (1.0 - 0.45 * sh_lift_need) + 0.04 * low_key * (1.0 - 0.55 * hi_pressure) - 0.025 * high_key) * (0.35 + 0.65 * narrow_dr_guard), 4),
            contrast=round(0.085 + 0.085 * flatness - 0.03 * sh_lift_need, 4),
            sat=round(0.02 + 0.05 * chroma_headroom - 0.04 * chroma_guard, 4),
            shoulder=round(0.02 + 0.12 * hi_pressure, 4),
            toe=round(-0.05 - 0.08 * milky_black + 0.14 * sh_lift_need, 4),
            tint=round(0.002 + 0.024 * green_cast - 0.022 * magenta_cast, 4),
        )
        grade = dict(
            blacks=round(-0.06 - 0.07 * milky_black + 0.11 * sh_lift_need, 4),
            shadows=round(0.02 + 0.15 * sh_lift_need, 4),
            whites=round(0.035 * (1.0 - hi_pressure), 4),
            highlights=round(-0.08 - 0.18 * hi_pressure, 4),
            clarity=round(0.07 + 0.05 * max(flatness, 0.0) * narrow_dr_guard, 4),
            sharpen=0.12,
            skin_protect=round(0.18 + 0.22 * skin_present, 4),
            tone_curve={
                "rgb": [
                    [0.0, 0.0],
                    [0.24, round(0.24 - s_amp * (1.0 - 0.72 * sh_lift_need), 4)],
                    [0.50, 0.500],
                    [0.76, round(0.76 + s_amp * (1.0 - 0.55 * hi_pressure), 4)],
                    [1.0, 1.0],
                ]
            },
            hsl={
                "yellow": [-0.10, -0.10, 0.02],
                "green": [0.08, -0.12, -0.08],
                "blue": [-0.08, round(0.08 * sat_sc, 3), round(-0.14 * hue_sc, 3)],
            },
            cg_shadows=[224.0, round(0.08 * sh_cool_room * tone_body, 3), round(-0.015 * (1.0 - 0.80 * sh_lift_need), 4)],
            cg_highlights=[44.0, round(0.18 * hi_warm_room * tone_body, 3), round(0.015 * (1.0 - 0.50 * hi_pressure), 4)],
            cg_blending=0.70,
        )
    elif preset == "cinema":
        # Kodak Vision3 500T / 2383 theatrical DI print look:
        # Deep oil-teal shadows vs warm tungsten-amber highlights, rolled filmic shoulder,
        # subtle matte-velvet cinema toe on contrasty scenes, calm pine greens, natural skin, organic grain.
        split_scale = (0.90 + 0.25 * max(chroma_headroom, -0.3)) * tone_body
        toe_lift = round(0.014 * (1.0 - milky_black) * narrow_dr_guard, 4)
        hi_cap = round(1.0 - 0.014 * (1.0 - 0.5 * hi_pressure) * narrow_dr_guard, 4)
        offsets = dict(
            ev=round(-0.02 * (1.0 - 0.70 * low_key) + 0.05 * low_key * (1.0 - 0.55 * hi_pressure) - 0.05 * high_key, 4),
            contrast=round(0.04 + 0.07 * flatness - 0.02 * sh_lift_need, 4),
            sat=round(-0.01 + 0.045 * max(chroma_headroom, 0.0) - 0.04 * chroma_guard, 4),
            shoulder=round(0.14 + 0.10 * hi_pressure, 4),
            toe=round(0.03 - 0.08 * milky_black + 0.10 * sh_lift_need, 4),
            tint=round(0.020 * green_cast - 0.018 * magenta_cast, 4),
        )
        grade = dict(
            blacks=round(0.045 * (1.0 - milky_black) - 0.06 * milky_black + 0.05 * sh_lift_need, 4),
            highlights=round(-0.13 - 0.18 * hi_pressure, 4),
            shadows=round(0.04 + 0.12 * sh_lift_need, 4),
            grain=0.08,
            clarity=round(0.06 + 0.05 * max(flatness, 0.0) * narrow_dr_guard, 4),
            skin_protect=round(0.28 + 0.26 * skin_present, 4),
            tone_curve={"rgb": [[0.0, toe_lift], [0.22, round(0.210 + 0.010 * sh_lift_need, 4)], [0.50, 0.494], [0.78, 0.784], [1.0, hi_cap]]},
            hsl={
                "red": [round(0.06 * hue_sc, 3), round(0.16 * sat_sc, 3), -0.14],
                "orange": [0.02, round(0.06 * sat_sc, 3), round(0.08 + 0.06 * skin_present, 3)],
                "yellow": [round(-0.22 * hue_sc, 3), -0.16, -0.04],
                "green": [round(0.18 * hue_sc, 3), round(-0.24 - 0.08 * green_present, 3), -0.18],
                "blue": [round(-0.22 * hue_sc, 3), round(0.12 * sat_sc, 3), -0.14],
            },
            cg_shadows=[196.0, round(0.30 * sh_cool_room * split_scale, 3), round(-0.025 * (1.0 - 0.80 * sh_lift_need), 4)],
            cg_midtones=[28.0, round(0.05 * hi_warm_room * split_scale, 3), 0.0],
            cg_highlights=[35.0, round(0.26 * hi_warm_room * split_scale, 3), round(0.015 * (1.0 - 0.50 * hi_pressure), 4)],
            cg_balance=-0.08,
            cg_blending=0.62,
        )
    else:
        mix = spec.get("camera_matrix_mix")
        offsets = dict(spec["offsets"])
        grade = copy.deepcopy(spec["grade"])

    if film == "bw":
        mix = None
        for ck in ("vibrance", "skin_protect", "hsl", "cg_shadows", "cg_midtones", "cg_highlights", "cg_balance", "cg_blending"):
            grade.pop(ck, None)
    return dict(camera_matrix_mix=mix, offsets=offsets, grade=grade)


def _clamp_finite(v: float, lo: float, hi: float, name: str) -> float:
    f = float(v)
    if not np.isfinite(f):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return float(np.clip(f, lo, hi))


def _validate_wheel(val, name: str) -> tuple[float, float, float]:
    if val is None:
        return (0.0, 0.0, 0.0)
    if isinstance(val, dict):
        h = val.get("h", 0.0)
        s = val.get("s", 0.0)
        l = val.get("l", 0.0)
    else:
        if len(val) != 3:
            raise ValueError(f"{name} must have 3 elements (h, s, l)")
        h, s, l = val
    hf = float(h) % 360.0 if np.isfinite(float(h)) else 0.0
    sf = _clamp_finite(s, 0.0, 1.0, f"{name}.s")
    lf = _clamp_finite(l, -1.0, 1.0, f"{name}.l")
    return (hf, sf, lf)


def _validate_hsl(val) -> dict[str, tuple[float, float, float]]:
    if not val:
        return {}
    out: dict[str, tuple[float, float, float]] = {}
    for band, triple in val.items():
        if band not in HSL_BANDS:
            raise ValueError(f"unknown HSL band {band!r}; expected one of {HSL_BANDS}")
        if isinstance(triple, dict):
            h, s, l = triple.get("h", 0.0), triple.get("s", 0.0), triple.get("l", 0.0)
        else:
            if len(triple) != 3:
                raise ValueError(f"hsl[{band!r}] must have 3 elements (h, s, l)")
            h, s, l = triple
        hf = _clamp_finite(h, -1.0, 1.0, f"hsl.{band}.h")
        sf = _clamp_finite(s, -1.0, 1.0, f"hsl.{band}.s")
        lf = _clamp_finite(l, -1.0, 1.0, f"hsl.{band}.l")
        if hf != 0.0 or sf != 0.0 or lf != 0.0:
            out[band] = (hf, sf, lf)
    return out


@dataclass
class DarkroomGrade:
    """Post-render darkroom creative grading (operates in display sRGB [0, 1])."""

    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    vibrance: float = 0.0
    skin_protect: float = 0.0
    tone_curve: tuple = ()
    cg_shadows: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cg_midtones: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cg_highlights: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cg_balance: float = 0.0
    cg_blending: float = 0.5
    hsl: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    dehaze: float = 0.0
    clarity: float = 0.0
    sharpen: float = 0.0
    vignette: float = 0.0
    grain: float = 0.0

    def __post_init__(self):
        self.highlights = _clamp_finite(self.highlights, -1.0, 1.0, "highlights")
        self.shadows = _clamp_finite(self.shadows, -1.0, 1.0, "shadows")
        self.whites = _clamp_finite(self.whites, -1.0, 1.0, "whites")
        self.blacks = _clamp_finite(self.blacks, -1.0, 1.0, "blacks")
        self.vibrance = _clamp_finite(self.vibrance, -1.0, 1.0, "vibrance")
        self.skin_protect = _clamp_finite(self.skin_protect, 0.0, 1.0, "skin_protect")
        self.tone_curve = _validate_curve(self.tone_curve)
        self.cg_shadows = _validate_wheel(self.cg_shadows, "cg_shadows")
        self.cg_midtones = _validate_wheel(self.cg_midtones, "cg_midtones")
        self.cg_highlights = _validate_wheel(self.cg_highlights, "cg_highlights")
        self.cg_balance = _clamp_finite(self.cg_balance, -1.0, 1.0, "cg_balance")
        self.cg_blending = _clamp_finite(self.cg_blending, 0.0, 1.0, "cg_blending")
        self.hsl = _validate_hsl(self.hsl)
        self.dehaze = _clamp_finite(self.dehaze, -1.0, 1.0, "dehaze")
        self.clarity = _clamp_finite(self.clarity, -1.0, 1.0, "clarity")
        self.sharpen = _clamp_finite(self.sharpen, 0.0, 1.0, "sharpen")
        self.vignette = _clamp_finite(self.vignette, -1.0, 1.0, "vignette")
        self.grain = _clamp_finite(self.grain, 0.0, 1.0, "grain")

    def has_color_grading(self) -> bool:
        return any(w[1] > 0.0 or w[2] != 0.0 for w in (self.cg_shadows, self.cg_midtones, self.cg_highlights))

    def is_identity(self) -> bool:
        return not (
            self.highlights or self.shadows or self.whites or self.blacks
            or self.vibrance or self.skin_protect or self.tone_curve
            or self.has_color_grading() or self.hsl
            or self.dehaze or self.clarity or self.sharpen or self.vignette or self.grain
        )

    def to_dict(self) -> dict:
        d: dict = {}
        for k in ("highlights", "shadows", "whites", "blacks", "vibrance", "skin_protect",
                  "dehaze", "clarity", "sharpen", "vignette", "grain"):
            v = getattr(self, k)
            if v != 0.0:
                d[k] = v
        if self.tone_curve:
            d["tone_curve"] = {ch: [list(pt) for pt in pts] for ch, pts in self.tone_curve}
        if self.has_color_grading():
            if self.cg_shadows[1] > 0.0 or self.cg_shadows[2] != 0.0:
                d["cg_shadows"] = list(self.cg_shadows)
            if self.cg_midtones[1] > 0.0 or self.cg_midtones[2] != 0.0:
                d["cg_midtones"] = list(self.cg_midtones)
            if self.cg_highlights[1] > 0.0 or self.cg_highlights[2] != 0.0:
                d["cg_highlights"] = list(self.cg_highlights)
            if self.cg_balance != 0.0:
                d["cg_balance"] = self.cg_balance
            if self.cg_blending != 0.5:
                d["cg_blending"] = self.cg_blending
        if self.hsl:
            d["hsl"] = {k: list(v) for k, v in self.hsl.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "DarkroomGrade":
        if not d:
            return cls()
        if isinstance(d, cls):
            return d
        known = {
            "highlights", "shadows", "whites", "blacks", "vibrance", "skin_protect",
            "tone_curve", "cg_shadows", "cg_midtones", "cg_highlights", "cg_balance",
            "cg_blending", "hsl", "dehaze", "clarity", "sharpen", "vignette", "grain",
        }
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown darkroom grade parameters: {sorted(unknown)}")
        return cls(**d)


@dataclass
class Geometry:
    """Display-oriented geometry transform: rot90 -> flip -> straighten angle -> normalized crop."""

    rot90: int = 0
    angle: float = 0.0
    flip_h: bool = False
    flip_v: bool = False
    crop: tuple[float, float, float, float] | None = None  # (top, left, bottom, right) in [0, 1]

    def __post_init__(self):
        self.rot90 = int(self.rot90) % 4
        self.angle = _clamp_finite(self.angle, -15.0, 15.0, "angle")
        self.flip_h = bool(self.flip_h)
        self.flip_v = bool(self.flip_v)
        if self.crop is not None:
            if len(self.crop) != 4:
                raise ValueError("crop must be [top, left, bottom, right]")
            t, l, b, r = (float(x) for x in self.crop)
            if not all(np.isfinite(x) for x in (t, l, b, r)):
                raise ValueError("crop coordinates must be finite")
            t = max(0.0, min(0.98, t))
            l = max(0.0, min(0.98, l))
            b = max(t + 0.02, min(1.0, b))
            r = max(l + 0.02, min(1.0, r))
            if t <= 1e-4 and l <= 1e-4 and b >= 1.0 - 1e-4 and r >= 1.0 - 1e-4:
                self.crop = None
            else:
                self.crop = (t, l, b, r)

    def is_identity(self) -> bool:
        return self.rot90 == 0 and abs(self.angle) < 1e-4 and not self.flip_h and not self.flip_v and self.crop is None

    def has_non_rot90(self) -> bool:
        return abs(self.angle) >= 1e-4 or self.flip_h or self.flip_v or self.crop is not None

    def to_dict(self) -> dict:
        d: dict = {}
        if self.rot90:
            d["rot90"] = self.rot90
        if abs(self.angle) >= 1e-4:
            d["angle"] = round(self.angle, 3)
        if self.flip_h:
            d["flip_h"] = True
        if self.flip_v:
            d["flip_v"] = True
        if self.crop is not None:
            d["crop"] = [round(v, 5) for v in self.crop]
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "Geometry":
        if not d:
            return cls()
        if isinstance(d, cls):
            return d
        return cls(
            rot90=d.get("rot90", 0),
            angle=d.get("angle", 0.0),
            flip_h=d.get("flip_h", False),
            flip_v=d.get("flip_v", False),
            crop=tuple(d["crop"]) if d.get("crop") else None,
        )


_LUT3D_N = 21
_g3 = np.linspace(0.0, 1.0, _LUT3D_N, dtype=np.float32)
_bb3, _gg3, _rr3 = np.meshgrid(_g3, _g3, _g3, indexing="ij")
_LUT3D_GRID = np.stack([_rr3, _gg3, _bb3], axis=-1)


def _apply_lut1d(arr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Fast 4096-entry 1D LUT lookup (error < 1.2e-4, ~30 ms on 1600px)."""
    idx = (np.clip(arr, np.float32(0.0), np.float32(1.0)) * np.float32(_LUT1D_N - 1) + np.float32(0.5)).astype(np.int32)
    return lut[idx]


def _apply_zonal_and_curve_lut1d(e: np.ndarray, g: DarkroomGrade, tc: tuple, *, inplace: bool = False) -> np.ndarray:
    base = _LUT1D_X
    if g.highlights or g.shadows or g.whites or g.blacks:
        rp_z = RenderParams(highlights=g.highlights, shadows=g.shadows, whites=g.whites, blacks=g.blacks)
        base = _apply_zonal_tones(base.copy(), rp_z)
    ch_map = dict(tc) if tc else {}
    if "rgb" in ch_map:
        base = eval_monotone_spline(ch_map["rgb"], base)
    has_per_ch = any(k in ch_map for k in ("r", "g", "b"))
    luts = [eval_monotone_spline(ch_map[k], base) if k in ch_map else base for k in ("r", "g", "b")] if has_per_ch else None

    h = e.shape[0]
    if h * e.shape[1] <= 500_000 and not inplace:
        if not has_per_ch:
            return _apply_lut1d(e, base)
        out = np.empty_like(e)
        for idx in range(3):
            out[..., idx] = _apply_lut1d(e[..., idx], luts[idx])
        return out

    out = e if inplace else np.empty_like(e)
    scale = np.float32(_LUT1D_N - 1)
    half = np.float32(0.5)

    def _run(rr):
        r0, r1 = rr
        ch = e[r0:r1]
        if not has_per_ch:
            idx = (np.clip(ch, np.float32(0.0), np.float32(1.0)) * scale + half).astype(np.int32)
            out[r0:r1] = base[idx]
        else:
            dst = out[r0:r1]
            for c in range(3):
                idx = (np.clip(ch[..., c], np.float32(0.0), np.float32(1.0)) * scale + half).astype(np.int32)
                dst[..., c] = luts[c][idx]

    _run_row_chunks(h, _run, step=128)
    return out


def _apply_pointwise_exact(u: np.ndarray, g: DarkroomGrade, tc: tuple, film: str) -> np.ndarray:
    if g.highlights or g.shadows or g.whites or g.blacks:
        rp_z = RenderParams(highlights=g.highlights, shadows=g.shadows, whites=g.whites, blacks=g.blacks)
        u = _apply_zonal_tones(u, rp_z)
    if g.vibrance and film != "bw":
        u = _apply_vibrance_srgb(u, g.vibrance)
    if tc:
        u = _apply_tone_curve(u, tc)
    if g.hsl and film != "bw":
        u = _apply_hsl(u, g.hsl)
    if g.skin_protect and film != "bw":
        u = _apply_skin_protect(u, g.skin_protect)
    if g.has_color_grading():
        u = _apply_color_grading(u, g)
    return np.clip(u, np.float32(0.0), np.float32(1.0))


_LUT3D_HI_N = 33
_gh3 = np.linspace(0.0, 1.0, _LUT3D_HI_N, dtype=np.float32)
_rrh, _ggh, _bbh = np.meshgrid(_gh3, _gh3, _gh3, indexing="ij")
_LUT3D_HI_GRID = np.stack([_rrh, _ggh, _bbh], axis=-1)


def _pointwise_cache_key(g: DarkroomGrade, tc: tuple, film: str) -> tuple:
    return (
        g.highlights, g.shadows, g.whites, g.blacks, g.vibrance, g.skin_protect,
        g.cg_shadows, g.cg_midtones, g.cg_highlights, g.cg_balance, g.cg_blending,
        tuple(sorted(g.hsl.items())), tc, film,
    )


@lru_cache(maxsize=32)
def _build_pil_lut3d(key: tuple):
    from PIL import ImageFilter

    (hi, sh, wh, bl, vib, sp, cgs, cgm, cgh, cgb, cgbl, hsl, tc, film) = key
    g = DarkroomGrade(
        highlights=hi, shadows=sh, whites=wh, blacks=bl, vibrance=vib, skin_protect=sp,
        cg_shadows=cgs, cg_midtones=cgm, cg_highlights=cgh, cg_balance=cgb, cg_blending=cgbl, hsl=dict(hsl),
    )
    lut = _apply_pointwise_exact(_LUT3D_GRID, g, tc, film)
    return ImageFilter.Color3DLUT(_LUT3D_N, lut.reshape(-1).tolist())


@lru_cache(maxsize=16)
def _build_f32_lut3d(key: tuple) -> np.ndarray:
    (hi, sh, wh, bl, vib, sp, cgs, cgm, cgh, cgb, cgbl, hsl, tc, film) = key
    g = DarkroomGrade(
        highlights=hi, shadows=sh, whites=wh, blacks=bl, vibrance=vib, skin_protect=sp,
        cg_shadows=cgs, cg_midtones=cgm, cg_highlights=cgh, cg_balance=cgb, cg_blending=cgbl, hsl=dict(hsl),
    )
    lut = _apply_pointwise_exact(_LUT3D_HI_GRID, g, tc, film)
    return np.ascontiguousarray(lut.reshape(-1, 3), dtype=np.float32)


def _apply_pointwise_lut3d(u: np.ndarray, g: DarkroomGrade, tc: tuple, film: str, *, inplace: bool = False) -> np.ndarray:
    """Chunked multi-threaded 21x21x21 C-accelerated PIL Color3DLUT (~2.7-3.3x faster on preview)."""
    from PIL import Image

    filt = _build_pil_lut3d(_pointwise_cache_key(g, tc, film))
    h, w = u.shape[:2]
    out = u if inplace else np.empty_like(u)
    inv255 = np.float32(1.0 / 255.0)

    def _run(rr):
        r0, r1 = rr
        u8 = np.round(np.clip(u[r0:r1], np.float32(0.0), np.float32(1.0)) * np.float32(255.0)).astype(np.uint8)
        out[r0:r1] = np.asarray(Image.fromarray(u8, mode="RGB").filter(filt), dtype=np.float32) * inv255

    _run_row_chunks(h, _run, step=128)
    return out


def _apply_pointwise_lut3d_f32(u: np.ndarray, g: DarkroomGrade, tc: tuple, film: str, *, inplace: bool = False) -> np.ndarray:
    """Chunked multi-threaded 33x33x33 float32 trilinear LUT for full-res 16-bit export (low RAM, L2-friendly)."""
    lut_flat = _build_f32_lut3d(_pointwise_cache_key(g, tc, film))
    h, w = u.shape[:2]
    out = u if inplace else np.empty_like(u)
    n = _LUT3D_HI_N
    max_i = np.int32(n - 2)
    scale = np.float32(n - 1)
    s0 = np.int32(n * n)
    s1 = np.int32(n)

    def _run(rr):
        r0, r1 = rr
        ch = u[r0:r1]
        rf = np.clip(ch[..., 0].ravel(), np.float32(0.0), np.float32(1.0)) * scale
        gf = np.clip(ch[..., 1].ravel(), np.float32(0.0), np.float32(1.0)) * scale
        bf = np.clip(ch[..., 2].ravel(), np.float32(0.0), np.float32(1.0)) * scale

        ri = np.minimum(rf.astype(np.int32), max_i)
        gi = np.minimum(gf.astype(np.int32), max_i)
        bi = np.minimum(bf.astype(np.int32), max_i)

        dr = (rf - ri)[:, None]
        dg = (gf - gi)[:, None]
        db = (bf - bi)[:, None]

        idx000 = ri * s0 + gi * s1 + bi
        c00 = lut_flat[idx000]
        c00 = c00 + db * (lut_flat[idx000 + 1] - c00)
        idx010 = idx000 + s1
        c01 = lut_flat[idx010]
        c01 = c01 + db * (lut_flat[idx010 + 1] - c01)
        c0 = c00 + dg * (c01 - c00)

        idx100 = idx000 + s0
        c10 = lut_flat[idx100]
        c10 = c10 + db * (lut_flat[idx100 + 1] - c10)
        idx110 = idx100 + s1
        c11 = lut_flat[idx110]
        c11 = c11 + db * (lut_flat[idx110 + 1] - c11)
        c1 = c10 + dg * (c11 - c10)

        res = c0 + dr * (c1 - c0)
        np.clip(res, np.float32(0.0), np.float32(1.0), out=res)
        out[r0:r1] = res.reshape(r1 - r0, w, 3)

    _run_row_chunks(h, _run, step=64)
    return out


def _hsv_hue_unit_rgb(hue_deg: float) -> np.ndarray:
    """Zero-luma chromatic tint direction for a hue angle in degrees."""
    h = (hue_deg % 360.0) / 60.0
    c = 1.0
    x = 1.0 - abs(h % 2.0 - 1.0)
    if h < 1:
        rgb = np.array([c, x, 0.0], dtype=np.float32)
    elif h < 2:
        rgb = np.array([x, c, 0.0], dtype=np.float32)
    elif h < 3:
        rgb = np.array([0.0, c, x], dtype=np.float32)
    elif h < 4:
        rgb = np.array([0.0, x, c], dtype=np.float32)
    elif h < 5:
        rgb = np.array([x, 0.0, c], dtype=np.float32)
    else:
        rgb = np.array([c, 0.0, x], dtype=np.float32)
    y = float(np.dot(rgb, LUMA_F32))
    return rgb - np.float32(y)


def _apply_vibrance_srgb(e: np.ndarray, vibrance: float) -> np.ndarray:
    """Adaptive chroma boost weighted toward muted colours (in sRGB [0, 1])."""
    y = (e[..., 0] * LUMA_F32[0] + e[..., 1] * LUMA_F32[1] + e[..., 2] * LUMA_F32[2])[..., None]
    mx = np.maximum(np.maximum(e[..., 0], e[..., 1]), e[..., 2])
    mn = np.minimum(np.minimum(e[..., 0], e[..., 1]), e[..., 2])
    sat = (mx - mn) / np.maximum(mx, np.float32(1e-4))
    w = (np.float32(1.0) - np.clip(sat, np.float32(0.0), np.float32(1.0)))[..., None]
    gain = np.float32(1.0) + np.float32(0.65 * vibrance) * w
    return np.clip(y + (e - y) * gain, np.float32(0.0), np.float32(1.0))


def _apply_color_grading(e: np.ndarray, g: DarkroomGrade) -> np.ndarray:
    """3-way Shadows / Midtones / Highlights luma-preserving split-tone color grading (1D weight LUT)."""
    y = e[..., 0] * LUMA_F32[0] + e[..., 1] * LUMA_F32[1] + e[..., 2] * LUMA_F32[2]
    pivot = np.float32(0.5 + 0.25 * g.cg_balance)
    width = np.float32(0.16 + 0.36 * g.cg_blending)
    lo_a = np.float32(max(0.0, float(pivot - width)))
    hi_a = np.float32(min(1.0, float(pivot + width)))

    # Compute the combined 3-channel offset LUT as a function of luminance Y in [0, 1] -> (4096, 3)!
    x = _LUT1D_X
    t_hi = _smoothstep(x, lo_a, hi_a)
    w_sh = (np.float32(1.0) - t_hi) * _smoothstep(x, 0.0, 0.04)
    w_hi = t_hi * (np.float32(1.0) - _smoothstep(x, 0.96, 1.0))
    w_mid = np.power(np.maximum(np.float32(4.0) * x * (np.float32(1.0) - x), np.float32(0.0)),
                     np.float32(1.4 - 0.6 * g.cg_blending))

    delta_lut = np.zeros((_LUT1D_N, 3), dtype=np.float32)
    for wheel, weight, strength in (
        (g.cg_shadows, w_sh, 0.22),
        (g.cg_midtones, w_mid, 0.20),
        (g.cg_highlights, w_hi, 0.22),
    ):
        h_deg, sat, lum = wheel
        if sat > 0.0:
            delta_lut += weight[:, None] * (_hsv_hue_unit_rgb(h_deg) * np.float32(sat * strength))
        if lum != 0.0:
            delta_lut += (weight * np.float32(lum * 0.14))[:, None]

    idx = np.clip((y * np.float32(_LUT1D_N - 1)).astype(np.int32), 0, _LUT1D_N - 1)
    return np.clip(e + delta_lut[idx], np.float32(0.0), np.float32(1.0))


def _apply_hsl(e: np.ndarray, hsl: dict[str, tuple[float, float, float]]) -> np.ndarray:
    """8-band smooth raised-cosine HSL color mixer via 720-entry hue LUT + branchless HSV->RGB."""
    # 1. Build tiny 720-entry 1D LUTs for dh, ds, dl (< 0.1 ms)
    dh_lut = np.zeros(_HUE_LUT_N, dtype=np.float32)
    ds_lut = np.zeros(_HUE_LUT_N, dtype=np.float32)
    dl_lut = np.zeros(_HUE_LUT_N, dtype=np.float32)
    n = len(_HSL_CENTERS)
    for i, name in enumerate(HSL_BANDS):
        if name not in hsl:
            continue
        bh, bs, bl = hsl[name]
        c = float(_HSL_CENTERS[i])
        span_l = (c - float(_HSL_CENTERS[(i - 1) % n])) % 360.0
        span_r = (float(_HSL_CENTERS[(i + 1) % n]) - c) % 360.0
        diff = ((_HUE_LUT_DEG - np.float32(c) + np.float32(180.0)) % np.float32(360.0)) - np.float32(180.0)
        t = np.where(diff >= 0.0, diff / np.float32(span_r), -diff / np.float32(span_l))
        w = np.where(t < np.float32(1.0),
                     np.float32(0.5) * (np.float32(1.0) + np.cos(np.float32(math.pi) * t)),
                     np.float32(0.0))
        if bh:
            dh_lut += w * np.float32(bh * 28.0)
        if bs:
            ds_lut += w * np.float32(bs)
        if bl:
            dl_lut += w * np.float32(bl)

    # 2. Fast RGB -> HSV
    r, g, b = e[..., 0], e[..., 1], e[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    sat = delta / np.maximum(mx, np.float32(1e-5))
    safe_d = np.maximum(delta, np.float32(1e-6))

    hue = np.where(
        delta <= np.float32(1e-6),
        np.float32(0.0),
        np.where(
            mx == r,
            np.float32(60.0) * (((g - b) / safe_d) % np.float32(6.0)),
            np.where(
                mx == g,
                np.float32(60.0) * (((b - r) / safe_d) + np.float32(2.0)),
                np.float32(60.0) * (((r - g) / safe_d) + np.float32(4.0)),
            ),
        ),
    )

    h_idx = np.clip((hue * np.float32(_HUE_LUT_N / 360.0)).astype(np.int32), 0, _HUE_LUT_N - 1)
    # Smooth chroma gate protects neutral greys
    c_gate = np.clip((sat - np.float32(0.03)) * np.float32(1.0 / 0.11), np.float32(0.0), np.float32(1.0))
    c_gate = c_gate * c_gate * (np.float32(3.0) - np.float32(2.0) * c_gate)

    h_new = np.mod(hue + dh_lut[h_idx] * c_gate, np.float32(360.0)) * np.float32(1.0 / 60.0)
    s_new = np.clip(sat * (np.float32(1.0) + ds_lut[h_idx] * c_gate), np.float32(0.0), np.float32(1.0))
    v_new = np.clip(mx + dl_lut[h_idx] * c_gate * np.float32(0.32) * mx * (np.float32(1.0) - mx * np.float32(0.55)),
                    np.float32(0.0), np.float32(1.0))

    # 3. Branchless HSV -> RGB: f(n) = v * (1 - s * clip(min(k, 4-k), 0, 1)) where k = (n + h/60) % 6
    out = np.empty_like(e)
    vs = v_new * s_new
    for ch, n_off in ((0, np.float32(5.0)), (1, np.float32(3.0)), (2, np.float32(1.0))):
        k = np.mod(h_new + n_off, np.float32(6.0))
        t = np.clip(np.minimum(k, np.float32(4.0) - k), np.float32(0.0), np.float32(1.0))
        out[..., ch] = v_new - vs * t
    return np.clip(out, np.float32(0.0), np.float32(1.0))


def _fast_blur_from_rgb(e: np.ndarray, radius_px: int, mode: str) -> np.ndarray:
    """Fast O(N) 2-pass separable box blur on luma or min-RGB (4x/2x downsampled for speed & low RAM)."""
    from PIL import Image

    h, w = e.shape[:2]
    ds = 4 if (max(h, w) > 1600 and radius_px >= 20) else (2 if (max(h, w) > 800 and radius_px >= 8) else 1)
    es = e[::ds, ::ds] if ds > 1 else e
    if mode == "min":
        ch = np.minimum(np.minimum(es[..., 0], es[..., 1]), es[..., 2])
    else:
        ch = es[..., 0] * LUMA_F32[0] + es[..., 1] * LUMA_F32[1] + es[..., 2] * LUMA_F32[2]
    k = max(3, int(round(radius_px / ds)) | 1)
    b = uniform_filter(uniform_filter(ch, size=k, mode="nearest"), size=k, mode="nearest")
    if ds > 1:
        return np.asarray(Image.fromarray(b, mode="F").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
    return b


def _apply_dehaze(e: np.ndarray, strength: float, *, film: str = "color", inplace: bool = False) -> np.ndarray:
    """Chunked multi-threaded darkroom haze removal / atmospheric restoration (~10x faster, low RAM)."""
    s = np.float32(strength)
    h, w = e.shape[:2]
    out = e if inplace else np.empty_like(e)
    is_color = (film != "bw")

    if s > 0.0:
        veil = _fast_blur_from_rgb(e, max(7, int(min(h, w) * 0.035)), "min")
        sat_k = np.float32(1.0) + np.float32(0.18) * s
        s_75 = np.float32(0.75) * s

        def _run_pos(rr):
            r0, r1 = rr
            ec = e[r0:r1]
            vc = np.minimum(veil[r0:r1], np.float32(0.32))
            tc = np.maximum(np.float32(1.0) - s_75 * vc, np.float32(0.35))
            res = (ec - (np.float32(1.0) - tc)[..., None] * np.float32(0.18)) / tc[..., None]
            np.clip(res, np.float32(0.0), np.float32(1.0), out=res)
            if is_color:
                yc = (res[..., 0] * LUMA_F32[0] + res[..., 1] * LUMA_F32[1] + res[..., 2] * LUMA_F32[2])[..., None]
                res -= yc
                res *= sat_k
                res += yc
                np.clip(res, np.float32(0.0), np.float32(1.0), out=res)
            out[r0:r1] = res

        _run_row_chunks(h, _run_pos, step=128)
        return out

    lift = np.float32(-0.18 * strength)
    gain = np.float32(1.0) - lift * np.float32(0.85)
    sat_k = np.float32(1.0) + np.float32(0.20) * s

    def _run_neg(rr):
        r0, r1 = rr
        res = lift + gain * e[r0:r1]
        if is_color:
            yc = (res[..., 0] * LUMA_F32[0] + res[..., 1] * LUMA_F32[1] + res[..., 2] * LUMA_F32[2])[..., None]
            res -= yc
            res *= sat_k
            res += yc
        np.clip(res, np.float32(0.0), np.float32(1.0), out=res)
        out[r0:r1] = res

    _run_row_chunks(h, _run_neg, step=128)
    return out


def _apply_clarity(e: np.ndarray, clarity: float, *, inplace: bool = False) -> np.ndarray:
    """Midtone-weighted local contrast on luminance (proportional radius for preview/export parity)."""
    h, w = e.shape[:2]
    low = _fast_blur_from_rgb(e, max(5, int(round(min(h, w) * 0.022))), "luma")

    out = e if inplace else np.empty_like(e)
    coeff = np.float32(0.55 * clarity)

    def _run(rr):
        r0, r1 = rr
        ec = e[r0:r1]
        yc = ec[..., 0] * LUMA_F32[0] + ec[..., 1] * LUMA_F32[1] + ec[..., 2] * LUMA_F32[2]
        delta = coeff * (yc - low[r0:r1]) * (np.float32(4.0) * yc * (np.float32(1.0) - yc))
        if inplace:
            ec += delta[..., None]
            np.clip(ec, np.float32(0.0), np.float32(1.0), out=ec)
        else:
            res = ec + delta[..., None]
            np.clip(res, np.float32(0.0), np.float32(1.0), out=res)
            out[r0:r1] = res

    _run_row_chunks(h, _run, step=128)
    return out


def _apply_sharpen(e: np.ndarray, sharpen: float, *, inplace: bool = False) -> np.ndarray:
    """Chunked luminance-only unsharp mask with overlap halo damping (true in-place rolling halo when inplace=True)."""
    h, w = e.shape[:2]
    sigma = max(0.8, min(h, w) * 0.0012)
    pad = max(3, int(math.ceil(2.5 * sigma)))
    coeff = np.float32(1.25 * sharpen)

    if inplace:
        step = max(256, pad)
        prev_halo: np.ndarray | None = None
        for r0 in range(0, h, step):
            r1 = min(h, r0 + step)
            p0 = max(0, r0 - pad)
            p1 = min(h, r1 + pad)
            if prev_halo is not None and p0 < r0:
                ec_pad = np.concatenate([prev_halo, e[r0:p1]], axis=0)
            else:
                ec_pad = e[p0:p1]
            next_halo = e[max(0, r1 - pad):r1].copy() if r1 < h else None
            yp = ec_pad[..., 0] * LUMA_F32[0] + ec_pad[..., 1] * LUMA_F32[1] + ec_pad[..., 2] * LUMA_F32[2]
            blur_p = gaussian_filter(yp, sigma=sigma, mode="nearest", truncate=2.5)
            s0 = r0 - p0
            s1 = s0 + (r1 - r0)
            hp = yp[s0:s1] - blur_p[s0:s1]
            gate = _smoothstep(np.abs(hp), 0.003, 0.015)
            delta = coeff * hp * gate
            ec = e[r0:r1]
            ec += delta[..., None]
            np.clip(ec, np.float32(0.0), np.float32(1.0), out=ec)
            prev_halo = next_halo
        return e

    out = np.empty_like(e)

    def _run(rr):
        r0, r1 = rr
        p0 = max(0, r0 - pad)
        p1 = min(h, r1 + pad)
        ec_pad = e[p0:p1]
        yp = ec_pad[..., 0] * LUMA_F32[0] + ec_pad[..., 1] * LUMA_F32[1] + ec_pad[..., 2] * LUMA_F32[2]
        blur_p = gaussian_filter(yp, sigma=sigma, mode="nearest", truncate=2.5)
        s0 = r0 - p0
        s1 = s0 + (r1 - r0)
        hp = yp[s0:s1] - blur_p[s0:s1]
        gate = _smoothstep(np.abs(hp), 0.003, 0.015)
        delta = coeff * hp * gate
        res = e[r0:r1] + delta[..., None]
        np.clip(res, np.float32(0.0), np.float32(1.0), out=res)
        out[r0:r1] = res

    _run_row_chunks(h, _run, step=256)
    return out


def _apply_grain(e: np.ndarray, grain: float, *, inplace: bool = False) -> np.ndarray:
    """Chunked multi-threaded deterministic midtone-weighted monochrome organic film grain."""
    h, w = e.shape[:2]
    out = e if inplace else np.empty_like(e)
    amp = np.float32(0.08 * grain)

    def _run(rr):
        r0, r1 = rr
        pad_t = 1 if r0 > 0 else 0
        pad_b = 1 if r1 < h else 0
        rng = np.random.default_rng(20261121 + r0 * 131)
        noise = rng.standard_normal((r1 - r0 + pad_t + pad_b, w), dtype=np.float32)
        noise = uniform_filter(noise, size=3, mode="nearest")[pad_t : pad_t + (r1 - r0)]
        ec = e[r0:r1]
        yc = ec[..., 0] * LUMA_F32[0] + ec[..., 1] * LUMA_F32[1] + ec[..., 2] * LUMA_F32[2]
        w_mid = np.float32(0.25) + np.float32(3.0) * yc * (np.float32(1.0) - yc)
        delta = (noise * amp * w_mid)[..., None]
        if inplace:
            ec += delta
            np.clip(ec, np.float32(0.0), np.float32(1.0), out=ec)
        else:
            res = ec + delta
            np.clip(res, np.float32(0.0), np.float32(1.0), out=res)
            out[r0:r1] = res

    _run_row_chunks(h, _run, step=256)
    return out


def apply_vignette(e: np.ndarray, vignette: float, *, inplace: bool = False) -> np.ndarray:
    """Chunked multi-threaded smooth elliptical radial vignette (~12x faster, low RAM)."""
    if not vignette:
        return e
    h, w = e.shape[:2]
    yy2 = np.linspace(-1.0, 1.0, h, dtype=np.float32) ** 2
    xx2 = np.linspace(-1.0, 1.0, w, dtype=np.float32) ** 2
    out = e if inplace else np.empty_like(e)
    neg = vignette < 0.0
    v_k = np.float32(0.75 * vignette if neg else 0.35 * vignette)

    def _run(rr):
        r0, r1 = rr
        falloff = _smoothstep(np.sqrt(xx2[None, :] + yy2[r0:r1, None]), 0.45, 1.42)
        ec = e[r0:r1]
        if neg:
            gain = (np.float32(1.0) + v_k * falloff)[..., None]
            if inplace:
                ec *= gain
                np.clip(ec, np.float32(0.0), np.float32(1.0), out=ec)
            else:
                res = ec * gain
                np.clip(res, np.float32(0.0), np.float32(1.0), out=res)
                out[r0:r1] = res
        else:
            lift = (v_k * falloff)[..., None]
            res = ec + lift * (np.float32(1.0) - ec)
            np.clip(res, np.float32(0.0), np.float32(1.0), out=res)
            out[r0:r1] = res

    _run_row_chunks(h, _run, step=128)
    return out


def apply_darkroom_grade(
    display: np.ndarray,
    grade: DarkroomGrade | dict | None,
    *,
    film: str = "color",
    apply_vignette_here: bool = False,
    fast: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Apply DarkroomGrade to sRGB [0, 1] float32 display image. Strict no-op when grade is identity."""
    g = DarkroomGrade.from_dict(grade)
    if g.is_identity():
        return display
    tc = tuple((ch, pts) for ch, pts in g.tone_curve if film != "bw" or ch == "rgb") if g.tone_curve else ()
    has_1d = bool(g.highlights or g.shadows or g.whites or g.blacks or tc)
    has_3d = bool((g.vibrance and film != "bw") or (g.hsl and film != "bw")
                  or (g.skin_protect and film != "bw") or g.has_color_grading())
    if not (g.dehaze or has_3d or has_1d or g.clarity or g.sharpen or g.grain or (apply_vignette_here and g.vignette)):
        return display
    u = np.asarray(display, dtype=np.float32)
    owned = bool(inplace) or (u is not display)

    # 1. Spatial Dehaze first (if active)
    if g.dehaze:
        u = _apply_dehaze(u, g.dehaze, film=film, inplace=owned)
        owned = True

    # 2. Point-wise color/tone stack: zonal, vibrance, tone curve, HSL, skin protect, 3-way color grading
    is_large = (u.shape[0] * u.shape[1]) > 1_000_000
    if has_3d:
        if fast:
            u = _apply_pointwise_lut3d(u, g, tc, film, inplace=owned)
            owned = True
        elif is_large:
            u = _apply_pointwise_lut3d_f32(u, g, tc, film, inplace=owned)
            owned = True
        else:
            u = _apply_pointwise_exact(u, g, tc, film)
            owned = True
    elif has_1d:
        u = _apply_zonal_and_curve_lut1d(u, g, tc, inplace=owned) if (fast or is_large) else _apply_pointwise_exact(u, g, tc, film)
        owned = True

    # 8. Clarity & Sharpen
    if g.clarity:
        u = _apply_clarity(u, g.clarity, inplace=owned)
        owned = True
    if g.sharpen:
        u = _apply_sharpen(u, g.sharpen, inplace=owned)
        owned = True

    # 9. Film Grain
    if g.grain:
        u = _apply_grain(u, g.grain, inplace=owned)
        owned = True

    # 10. Optional Vignette (caller usually applies vignette after apply_geometry so crop is respected)
    if apply_vignette_here and g.vignette:
        u = apply_vignette(u, g.vignette, inplace=owned)
        owned = True

    return u if owned else np.clip(u, np.float32(0.0), np.float32(1.0))


def _inscribed_scale(h: int, w: int, angle_deg: float) -> float:
    """Minimum zoom factor (>= 1) so rotating a (h, w) image by angle_deg leaves zero empty corners."""
    a = abs(math.radians(angle_deg))
    if a < 1e-6:
        return 1.0
    ca, sa = math.cos(a), math.sin(a)
    return max(ca + sa * (w / max(h, 1)), ca + sa * (h / max(w, 1)))


def apply_geometry(
    img: np.ndarray,
    geom: Geometry | dict | None,
    *,
    apply_rot90: bool = True,
    apply_crop: bool = True,
    fast_preview: bool = False,
) -> np.ndarray:
    """Apply rot90, flip, straighten angle (with auto inscribed zoom), and normalized crop."""
    gm = Geometry.from_dict(geom)
    out = img
    if apply_rot90 and gm.rot90:
        out = np.rot90(out, gm.rot90)
    if gm.flip_h:
        out = out[:, ::-1]
    if gm.flip_v:
        out = out[::-1, :]
    if abs(gm.angle) >= 1e-4:
        h, w = out.shape[:2]
        scale = _inscribed_scale(h, w, gm.angle)
        rad = math.radians(gm.angle)
        ca, sa = math.cos(rad) / scale, math.sin(rad) / scale
        if fast_preview:
            from PIL import Image
            # PIL affine evaluates at pixel centers (u + 0.5, v + 0.5) and subtracts 0.5 on sample
            cx_pil, cy_pil = w * 0.5, h * 0.5
            # Matches scipy row/col convention: row' = ca*r - sa*c, col' = sa*r + ca*c
            a_m, b_m = ca, sa
            d_m, e_m = -sa, ca
            c_m = cx_pil - a_m * cx_pil - b_m * cy_pil
            f_m = cy_pil - d_m * cx_pil - e_m * cy_pil
            u8_f = np.clip(np.asarray(out, dtype=np.float32), np.float32(0.0), np.float32(1.0))
            u8_f *= np.float32(255.0)
            u8_f += np.float32(0.5)
            u8 = u8_f.astype(np.uint8)
            if u8.ndim == 2:
                im = Image.fromarray(u8, mode="L").transform(
                    (w, h), Image.Transform.AFFINE, (a_m, b_m, c_m, d_m, e_m, f_m), resample=Image.Resampling.BILINEAR
                )
                out = np.asarray(im, dtype=np.float32) * np.float32(1.0 / 255.0)
            elif u8.ndim == 3 and u8.shape[2] == 1:
                im = Image.fromarray(u8[..., 0], mode="L").transform(
                    (w, h), Image.Transform.AFFINE, (a_m, b_m, c_m, d_m, e_m, f_m), resample=Image.Resampling.BILINEAR
                )
                out = (np.asarray(im, dtype=np.float32) * np.float32(1.0 / 255.0))[..., None]
            else:
                im = Image.fromarray(u8, mode="RGB").transform(
                    (w, h), Image.Transform.AFFINE, (a_m, b_m, c_m, d_m, e_m, f_m), resample=Image.Resampling.BILINEAR
                )
                out = np.asarray(im, dtype=np.float32) * np.float32(1.0 / 255.0)
        else:
            mat = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
            center = np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=np.float64)
            offset = center - mat @ center
            if out.ndim == 2:
                src_2d = np.ascontiguousarray(out, dtype=np.float32)
                rot_out = np.empty((h, w), dtype=np.float32)
                def _rot_chunk_2d(rr):
                    r0, r1 = rr
                    off_r = offset + mat[:, 0] * r0
                    dst_c = np.empty((r1 - r0, w), dtype=np.float32)
                    affine_transform(src_2d, mat, offset=off_r, output_shape=(r1 - r0, w),
                                     output=dst_c, order=1, mode="nearest", prefilter=False)
                    np.clip(dst_c, np.float32(0.0), np.float32(1.0), out=dst_c)
                    rot_out[r0:r1, :] = dst_c
                _run_row_chunks(h, _rot_chunk_2d, step=256)
                out = rot_out
            else:
                n_ch = out.shape[2]
                rot_out = np.empty((h, w, n_ch), dtype=np.float32)
                for c in range(n_ch):
                    src_c = np.ascontiguousarray(out[..., c], dtype=np.float32)
                    def _rot_chunk(rr, _src=src_c, _c=c):
                        r0, r1 = rr
                        off_r = offset + mat[:, 0] * r0
                        dst_c = np.empty((r1 - r0, w), dtype=np.float32)
                        affine_transform(_src, mat, offset=off_r, output_shape=(r1 - r0, w),
                                         output=dst_c, order=1, mode="nearest", prefilter=False)
                        np.clip(dst_c, np.float32(0.0), np.float32(1.0), out=dst_c)
                        rot_out[r0:r1, :, _c] = dst_c
                    _run_row_chunks(h, _rot_chunk, step=256)
                    del src_c
                out = rot_out
    if apply_crop and gm.crop is not None:
        t, l, b, r = gm.crop
        h, w = out.shape[:2]
        y0 = max(0, min(h - 2, int(round(t * h))))
        x0 = max(0, min(w - 2, int(round(l * w))))
        y1 = max(y0 + 1, min(h, int(round(b * h))))
        x1 = max(x0 + 1, min(w, int(round(r * w))))
        out = np.ascontiguousarray(out[y0:y1, x0:x1])
    elif out is not img and not out.flags.c_contiguous:
        out = np.ascontiguousarray(out)
    return out


def apply_lr_crop_angle(
    img: np.ndarray,
    crop: tuple[float, float, float, float] | None,
    angle: float = 0.0,
    *,
    fast_preview: bool = False,
) -> np.ndarray:
    """Apply Adobe Lightroom / Camera Raw (CropTop, CropLeft, CropBottom, CropRight, CropAngle).

    In Lightroom's XMP / catalog geometry (in raw sensor space before 90-deg orientation):
      - (CropLeft * W, CropTop * H) is the top-left corner of the rotated crop box
        in source image coordinates; (CropRight * W, CropBottom * H) is its bottom-right.
      - The rotated crop box's width/height are:
            dx = (CropRight - CropLeft) * W,  dy = (CropBottom - CropTop) * H
            cw =  dx * cos(th) + dy * sin(th)
            ch = -dx * sin(th) + dy * cos(th)   where th = radians(CropAngle)
      - Inverse mapping from output pixel (u, v) to source pixel (x_src, y_src):
            x_src = cx_src + (u - cx_dst) * cos(th) - (v - cy_dst) * sin(th)
            y_src = cy_src + (u - cx_dst) * sin(th) + (v - cy_dst) * cos(th)
    """
    ang = float(angle or 0.0)
    if not np.isfinite(ang):
        raise ValueError(f"angle must be finite, got {angle!r}")
    if crop is not None:
        if len(crop) != 4:
            raise ValueError(f"crop must be a 4-tuple (top, left, bottom, right), got {crop!r}")
        t, l, b, r = (float(v) for v in crop)
        if not all(np.isfinite(v) for v in (t, l, b, r)):
            raise ValueError(f"crop values must be finite, got {crop!r}")
        if b <= t or r <= l:
            raise ValueError(f"crop must satisfy bottom > top and right > left, got {crop!r}")
        crop = (t, l, b, r)
    if crop is None and abs(ang) < 1e-4:
        return img
    h, w = img.shape[:2]

    # Axis-aligned crop only (no rotation): exact fast slice
    if crop is not None and abs(ang) < 1e-4:
        t, l, b, r = crop
        y0 = max(0, min(h - 2, int(round(t * h))))
        x0 = max(0, min(w - 2, int(round(l * w))))
        y1 = max(y0 + 1, min(h, int(round(b * h))))
        x1 = max(x0 + 1, min(w, int(round(r * w))))
        return np.ascontiguousarray(img[y0:y1, x0:x1])

    th = math.radians(ang)
    ct, st = math.cos(th), math.sin(th)

    if crop is not None:
        t, l, b, r = crop
        dx = (r - l) * w
        dy = (b - t) * h
        cw = dx * ct + dy * st
        ch = -dx * st + dy * ct
        if cw < 2.0 or ch < 2.0:
            y0 = max(0, min(h - 2, int(round(t * h))))
            x0 = max(0, min(w - 2, int(round(l * w))))
            y1 = max(y0 + 1, min(h, int(round(b * h))))
            x1 = max(x0 + 1, min(w, int(round(r * w))))
            return np.ascontiguousarray(img[y0:y1, x0:x1])
        out_w = max(2, int(round(cw)))
        out_h = max(2, int(round(ch)))
        cx_src = 0.5 * (l + r) * w - 0.5
        cy_src = 0.5 * (t + b) * h - 0.5
        cx_dst = (out_w - 1) / 2.0
        cy_dst = (out_h - 1) / 2.0
        ca, sa = ct, st
    else:
        # Angle only (no crop box): inscribed zoom around image center with LR sign
        scale = _inscribed_scale(h, w, ang)
        out_w, out_h = w, h
        cx_src = (w - 1) / 2.0
        cy_src = (h - 1) / 2.0
        cx_dst, cy_dst = cx_src, cy_src
        ca, sa = ct / scale, st / scale

    if fast_preview:
        from PIL import Image
        a_m, b_m = ca, -sa
        d_m, e_m = sa, ca
        cx_src_pil, cy_src_pil = cx_src + 0.5, cy_src + 0.5
        cx_dst_pil, cy_dst_pil = cx_dst + 0.5, cy_dst + 0.5
        c_m = cx_src_pil - a_m * cx_dst_pil - b_m * cy_dst_pil
        f_m = cy_src_pil - d_m * cx_dst_pil - e_m * cy_dst_pil
        u8_f = np.clip(np.asarray(img, dtype=np.float32), np.float32(0.0), np.float32(1.0))
        u8_f *= np.float32(255.0)
        u8_f += np.float32(0.5)
        u8 = u8_f.astype(np.uint8)
        if u8.ndim == 2:
            im = Image.fromarray(u8, mode="L").transform(
                (out_w, out_h), Image.Transform.AFFINE, (a_m, b_m, c_m, d_m, e_m, f_m),
                resample=Image.Resampling.BILINEAR,
            )
            return np.asarray(im, dtype=np.float32) * np.float32(1.0 / 255.0)
        if u8.ndim == 3 and u8.shape[2] == 1:
            im = Image.fromarray(u8[..., 0], mode="L").transform(
                (out_w, out_h), Image.Transform.AFFINE, (a_m, b_m, c_m, d_m, e_m, f_m),
                resample=Image.Resampling.BILINEAR,
            )
            return (np.asarray(im, dtype=np.float32) * np.float32(1.0 / 255.0))[..., None]
        im = Image.fromarray(u8, mode="RGB").transform(
            (out_w, out_h), Image.Transform.AFFINE, (a_m, b_m, c_m, d_m, e_m, f_m),
            resample=Image.Resampling.BILINEAR,
        )
        return np.asarray(im, dtype=np.float32) * np.float32(1.0 / 255.0)

    # Full-precision float32 SciPy affine_transform in (row, col) = (y, x) coordinates:
    #   y_src =  ca * (v - cy_dst) + sa * (u - cx_dst) + cy_src
    #   x_src = -sa * (v - cy_dst) + ca * (u - cx_dst) + cx_src
    mat = np.array([[ca, sa], [-sa, ca]], dtype=np.float64)
    offset = np.array([cy_src, cx_src], dtype=np.float64) - mat @ np.array([cy_dst, cx_dst], dtype=np.float64)
    if img.ndim == 2:
        src_2d = np.ascontiguousarray(img, dtype=np.float32)
        rot_out = np.empty((out_h, out_w), dtype=np.float32)
        def _rot_chunk_2d(rr):
            r0, r1 = rr
            off_r = offset + mat[:, 0] * r0
            dst_c = np.empty((r1 - r0, out_w), dtype=np.float32)
            affine_transform(src_2d, mat, offset=off_r, output_shape=(r1 - r0, out_w),
                             output=dst_c, order=1, mode="nearest", prefilter=False)
            np.clip(dst_c, np.float32(0.0), np.float32(1.0), out=dst_c)
            rot_out[r0:r1, :] = dst_c
        _run_row_chunks(out_h, _rot_chunk_2d, step=256)
        return rot_out

    n_ch = img.shape[2]
    rot_out = np.empty((out_h, out_w, n_ch), dtype=np.float32)
    for c in range(n_ch):
        src_c = np.ascontiguousarray(img[..., c], dtype=np.float32)
        def _rot_chunk(rr, _src=src_c, _c=c):
            r0, r1 = rr
            off_r = offset + mat[:, 0] * r0
            dst_c = np.empty((r1 - r0, out_w), dtype=np.float32)
            affine_transform(_src, mat, offset=off_r, output_shape=(r1 - r0, out_w),
                             output=dst_c, order=1, mode="nearest", prefilter=False)
            np.clip(dst_c, np.float32(0.0), np.float32(1.0), out=dst_c)
            rot_out[r0:r1, :, _c] = dst_c
        _run_row_chunks(out_h, _rot_chunk, step=256)
        del src_c
    return rot_out


def free_stem_ext(out_dir: Path, stem: str, ext: str, *, check_json: bool = True) -> str:
    """Non-overwriting stem finder for arbitrary extension ('.tif', '.jpg', '.png')."""
    out_dir = Path(out_dir)
    name, n = stem, 1
    while (out_dir / f"{name}{ext}").exists() or (check_json and (out_dir / f"{name}.json").exists()):
        n += 1
        name = f"{stem}-{n}"
    return name


def export_image(
    path: Path,
    img_float: np.ndarray,
    *,
    fmt: str = "tiff16",
    encoding: str = "srgb",
    quality: int = 95,
    max_edge: int | None = None,
    fast_tiff: bool = False,
) -> Path:
    """Write float32 [0, 1] RGB image to 16-bit TIFF, 8-bit JPEG, or 8-bit PNG with embedded ICC profile."""
    from PIL import Image

    if fmt not in ("tiff16", "jpeg", "png"):
        raise ValueError(f"unsupported export format {fmt!r}; expected 'tiff16', 'jpeg', or 'png'")
    if encoding not in DESCRIPTION:
        raise ValueError(f"unsupported encoding {encoding!r}")

    arr = np.asarray(img_float, dtype=np.float32)
    h, w = arr.shape[:2]
    will_resize = max_edge is not None and int(max_edge) > 0 and max(h, w) > int(max_edge)
    if fmt == "tiff16" and not fast_tiff and not will_resize:
        arr = np.clip(arr, np.float32(0.0), np.float32(1.0))

    if will_resize:
        scale = int(max_edge) / float(max(h, w))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        if fmt == "tiff16":
            chans = [
                np.asarray(Image.fromarray(arr[..., c], mode="F").resize((nw, nh), Image.Resampling.LANCZOS), dtype=np.float32)
                for c in range(3)
            ]
            arr = np.clip(np.stack(chans, axis=-1), 0.0, 1.0)
            im = None
        else:
            u8 = np.clip(arr * np.float32(255.0) + np.float32(0.5), np.float32(0.0), np.float32(255.0)).astype(np.uint8)
            im = Image.fromarray(u8, mode="RGB").resize((nw, nh), Image.Resampling.LANCZOS)
            arr = None
    else:
        im = None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "tiff16":
        if fast_tiff:
            import tifffile
            u16 = np.empty(arr.shape, dtype=np.uint16)
            def _to_u16(rr):
                r0, r1 = rr
                ch = np.clip(arr[r0:r1], np.float32(0.0), np.float32(1.0))
                ch *= np.float32(65535.0)
                ch += np.float32(0.5)
                u16[r0:r1] = ch.astype(np.uint16)
            _run_row_chunks(arr.shape[0], _to_u16, step=256)
            tifffile.imwrite(
                path, u16, photometric="rgb", iccprofile=icc_profile(encoding),
                compression=None, software="filmcolor",
            )
            return path
        write_tiff16(path, arr, encoding)
        return path

    if im is None:
        u8 = np.clip(arr * np.float32(255.0) + np.float32(0.5), np.float32(0.0), np.float32(255.0)).astype(np.uint8)
        im = Image.fromarray(u8, mode="RGB")
    icc = icc_profile(encoding)
    if fmt == "jpeg":
        q = int(np.clip(int(quality), 60, 100))
        im.save(path, format="JPEG", quality=q, subsampling=0 if q >= 92 else 2, optimize=True, icc_profile=icc)
    else:
        im.save(path, format="PNG", compress_level=6, icc_profile=icc)
    return path

# T25 final release preset alias
T25_STYLE_PRESETS = T24_STYLE_PRESETS
