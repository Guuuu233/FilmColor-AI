"""Darkroom post-grading, geometry (crop/straighten/flip), style presets, and multi-format export.

Pure NumPy + SciPy + Pillow. All operations are strict no-ops when at default values so existing
pinned RenderParams / RenderParamsT24 outputs remain bit-identical unless post-grading is active.
"""
from __future__ import annotations

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
        desc="自适应 SP-3000 鲜润影调：通透翡翠绿与清亮肤色，按画面反差与色度余量动态调优（日常直出推荐）",
        camera_matrix_mix=0.58,
        offsets=dict(sat=0.12, contrast=0.08, shoulder=0.0, toe=-0.16, tint=0.018, temp=0.008),
        grade=dict(
            vibrance=0.14, highlights=-0.10, blacks=-0.05, clarity=0.05,
            hsl={"green": [-0.08, 0.12, -0.03], "orange": [0.0, 0.06, 0.04]},
        ),
    ),
    "leica": dict(
        label="德味徕卡 Leica Look",
        desc="自适应德味大师影调：油润厚重红蓝、克制松木绿、强立体微反差、冷暗暖亮分离与轻光学暗角",
        camera_matrix_mix=0.58,
        offsets=dict(ev=-0.06, contrast=0.09, sat=0.05, shoulder=0.07, toe=-0.18, tint=0.014),
        grade=dict(
            vibrance=0.09, highlights=-0.12, shadows=0.05, blacks=-0.07,
            clarity=0.12, dehaze=0.06, sharpen=0.10, vignette=-0.11, skin_protect=0.26,
            tone_curve={"rgb": [[0.0, 0.0], [0.22, 0.195], [0.50, 0.492], [0.78, 0.805], [1.0, 1.0]]},
            hsl={
                "red": [-0.05, 0.14, -0.09],
                "orange": [0.02, 0.01, 0.05],
                "yellow": [-0.10, -0.08, -0.06],
                "green": [0.08, -0.12, -0.10],
                "aqua": [0.0, -0.08, -0.06],
                "blue": [-0.05, 0.10, -0.13],
            },
            cg_shadows=[218.0, 0.09, -0.04],
            cg_highlights=[43.0, 0.08, 0.02],
            cg_blending=0.68,
        ),
    ),
    "noritsu": dict(
        label="诺日士 Noritsu 通透",
        desc="自适应 HS-1800 通透店扫：高光柔顺、暗部清澈开放、象牙白干净且肤色自然克制",
        camera_matrix_mix=0.54,
        offsets=dict(sat=0.08, contrast=0.04, shoulder=0.06, toe=-0.08, tint=0.014, temp=0.006),
        grade=dict(
            vibrance=0.10, highlights=-0.12, shadows=0.06, blacks=-0.03, skin_protect=0.25,
            cg_highlights=[46.0, 0.05, 0.01], cg_blending=0.72,
        ),
    ),
    "linear": dict(
        label="线性 Linear",
        desc="自适应高宽容平坦底稿：最大化保留高光与暗部动态范围、中性色度且无风格化染调（专为 Lr/PS 深度自调设计）",
        camera_matrix_mix=0.50,
        offsets=dict(contrast=-0.20, sat=-0.08, shoulder=0.20, toe=0.08, tint=0.010),
        grade=dict(highlights=-0.18, shadows=0.16, blacks=0.03),
    ),
    "standard": dict(
        label="标准 Standard",
        desc="最新 T24 锁定物理基准，无额外风格偏移（色度最保守）",
        offsets=dict(),
        grade=dict(),
    ),
    "natural": dict(
        label="自然 Natural",
        desc="自适应温润胶片影调：柔和光比、开放暗部与细腻暖白高光，耐看无侵略性",
        offsets=dict(contrast=-0.06, sat=-0.02, shoulder=0.08),
        grade=dict(shadows=0.08, highlights=-0.08, vibrance=0.05, cg_highlights=[42.0, 0.06, 0.01], cg_blending=0.72),
    ),
    "portrait": dict(
        label="人像 Portrait",
        desc="自适应 Portra 柔美肤色：通透蜜桃肌理、柔化绿植反光、柔和光比与动态肤色保护",
        offsets=dict(ev=0.07, contrast=-0.08, sat=-0.03, shoulder=0.10, temp=0.010, tint=0.014),
        grade=dict(
            shadows=0.10, highlights=-0.12, skin_protect=0.45, vibrance=0.06,
            hsl={"orange": [0.02, -0.02, 0.06], "yellow": [-0.08, -0.08, 0.03], "green": [0.04, -0.10, -0.02]},
        ),
    ),
    "vivid": dict(
        label="鲜明 Vivid",
        desc="自适应反转片 Velvia 浓郁质感：深邃天空与饱和红橙、通透微反差，强饱和场景自动防溢出",
        camera_matrix_mix=0.58,
        offsets=dict(contrast=0.11, sat=0.10, shoulder=-0.02),
        grade=dict(
            vibrance=0.16, highlights=-0.08, blacks=-0.06, clarity=0.08,
            hsl={"red": [-0.03, 0.12, -0.05], "blue": [-0.04, 0.10, -0.10]},
        ),
    ),
    "print_classic": dict(
        label="经典印相 Print",
        desc="自适应光学暗房相纸：随底片反差动态调节 S 曲线，扎实黑位、温润纸白与细腻锐度",
        offsets=dict(contrast=0.09, sat=0.02, shoulder=-0.04),
        grade=dict(
            blacks=-0.07, whites=0.03, sharpen=0.10,
            tone_curve={"rgb": [[0.0, 0.0], [0.24, 0.21], [0.5, 0.50], [0.76, 0.79], [1.0, 1.0]]},
            cg_highlights=[45.0, 0.07, 0.01], cg_blending=0.70,
        ),
    ),
    "cinema": dict(
        label="电影质感 Cinema",
        desc="自适应 Vision3 银幕色调：油青暗部与钨丝暖高光分离、克制绿植与细腻有机颗粒",
        offsets=dict(contrast=0.04, sat=-0.02, shoulder=0.12),
        grade=dict(
            blacks=0.05, highlights=-0.10, grain=0.07,
            hsl={"green": [0.06, -0.12, -0.06]},
            cg_shadows=[198.0, 0.14, -0.03],
            cg_highlights=[36.0, 0.12, 0.02],
            cg_blending=0.65,
            clarity=0.04,
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
    arr = np.clip(np.asarray(display, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    h, w = arr.shape[:2]
    step = max(1, max(h, w) // 192)
    sub = arr[::step, ::step]
    if mask is not None and mask.shape[:2] == (h, w):
        m = np.asarray(mask[::step, ::step], dtype=bool)
        px = sub[m] if int(m.sum()) >= 64 else sub.reshape(-1, 3)
    else:
        px = sub.reshape(-1, 3)
    if px.size == 0:
        return dict(
            luma_p02=0.04, luma_p05=0.08, luma_p50=0.45, luma_p95=0.82, luma_p98=0.90,
            dr=0.74, hi_clip=0.0, sh_crush=0.02, chroma_p50=0.25, chroma_p90=0.45,
            skin_share=0.0, blue_share=0.0, warm_bias=0.0, green_bias=0.0,
        )

    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    y = r * LUMA_F32[0] + g * LUMA_F32[1] + b * LUMA_F32[2]
    p02, p05, p50, p95, p98 = (float(v) for v in np.percentile(y, [2, 5, 50, 95, 98]))

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

    # Sky / water blue presence
    blue_share = float(np.mean((b > g) & (b > r) & (sat >= np.float32(0.14)) & (y >= np.float32(0.22))))

    return dict(
        luma_p02=p02, luma_p05=p05, luma_p50=p50, luma_p95=p95, luma_p98=p98,
        dr=max(0.0, p95 - p05), hi_clip=hi_clip, sh_crush=sh_crush,
        chroma_p50=c50, chroma_p90=c90, skin_share=skin_share, blue_share=blue_share,
        warm_bias=warm_bias, green_bias=green_bias,
    )


def resolve_adaptive_preset(
    preset: str,
    film: str = "color",
    scene_metrics: dict[str, float] | None = None,
) -> dict:
    """Resolve preset parameters adaptively from scene_metrics (falls back cleanly to T24_STYLE_PRESETS when None)."""
    spec = T24_STYLE_PRESETS.get(preset) or T24_STYLE_PRESETS["standard"]
    if preset == "standard" or scene_metrics is None:
        return dict(
            camera_matrix_mix=spec.get("camera_matrix_mix"),
            offsets=dict(spec["offsets"]),
            grade=dict(spec["grade"]),
        )

    m = scene_metrics
    dr = float(m.get("dr", 0.72))
    p02 = float(m.get("luma_p02", 0.04))
    p05 = float(m.get("luma_p05", 0.08))
    p50 = float(m.get("luma_p50", 0.45))
    p98 = float(m.get("luma_p98", 0.90))
    hi_clip = float(m.get("hi_clip", 0.0))
    sh_crush = float(m.get("sh_crush", 0.0))
    c90 = float(m.get("chroma_p90", 0.45))
    skin_share = float(m.get("skin_share", 0.0))
    blue_share = float(m.get("blue_share", 0.0))
    warm_bias = float(m.get("warm_bias", 0.0))
    green_bias = float(m.get("green_bias", 0.0))

    # Smooth normalized adaptive drivers (calibrated on 294-negative / 8-camera-format benchmark)
    flatness = float(np.clip((0.72 - dr) / 0.28, -0.60, 1.0))
    hi_pressure = float(np.clip((p98 - 0.87) / 0.12, 0.0, 1.0) * 0.55 + np.clip(hi_clip / 0.12, 0.0, 1.0) * 0.45)
    sh_lift_need = float(np.clip(max((0.10 - p05) / 0.08, sh_crush / 0.16), 0.0, 1.0))
    milky_black = float(np.clip((p05 - 0.11) / 0.12, 0.0, 1.0) * np.clip((p50 - 0.32) / 0.16, 0.20, 1.0))
    veil_need = float(np.clip((min(p02, p05) - 0.07) / 0.14, 0.0, 1.0))
    chroma_headroom = float(np.clip((0.46 - c90) / 0.28, -1.0, 1.0))
    chroma_guard = float(np.clip((c90 - 0.52) / 0.32, 0.0, 1.0))
    skin_present = float(np.clip((skin_share - 0.03) / 0.15, 0.0, 1.0))
    blue_present = float(np.clip((blue_share - 0.04) / 0.20, 0.0, 1.0))
    green_cast = float(np.clip(green_bias / 0.035, 0.0, 1.0))
    magenta_cast = float(np.clip(-green_bias / 0.035, 0.0, 1.0))
    cool_need = float(np.clip(-warm_bias / 0.10, -0.5, 1.0))
    warm_excess = float(np.clip((warm_bias - 0.06) / 0.14, 0.0, 1.0))
    dark_mid = float(np.clip((0.48 - p50) / 0.20, -0.5, 1.0))
    bright_mid = float(np.clip((p50 - 0.62) / 0.16, 0.0, 1.0))

    mix: float | None = None
    offsets: dict[str, float] = {}
    grade: dict = {}

    if preset == "leica":
        # Authentic German-glass / Leica Look: sculptural micro-contrast, oil-rich reds & cobalt blues,
        # restrained pine-olive greens, anchored velvety blacks on normal/milky frames with shadow-crush guard
        # on low-key scenes, creamy ivory highlights, bidirectional neutral tint, adaptive EV centering.
        mix = round(0.50 + 0.06 * (1.0 - chroma_guard) + 0.06 * max(chroma_headroom, 0.0), 3)
        ev_adj = round(0.03 * (1.0 - hi_pressure) + 0.14 * max(dark_mid, 0.0) * (1.0 - 0.55 * hi_pressure) - 0.10 * bright_mid, 4)
        s_low = round(0.22 - (0.024 + 0.016 * max(flatness, -0.35)) * (1.0 - 0.78 * sh_lift_need), 4)
        s_hi = round(0.78 + (0.022 + 0.012 * max(flatness, -0.35)) * (1.0 - 0.55 * hi_pressure), 4)
        hsl_c = 1.0 - 0.85 * chroma_guard
        offsets = dict(
            ev=ev_adj,
            contrast=round(0.07 + 0.09 * flatness - 0.03 * sh_lift_need, 4),
            sat=round(0.03 + 0.07 * chroma_headroom - 0.04 * chroma_guard, 4),
            shoulder=round(0.06 + 0.11 * hi_pressure, 4),
            toe=round(-0.13 - 0.14 * milky_black + 0.19 * sh_lift_need, 4),
            tint=round(0.003 + 0.028 * green_cast - 0.022 * magenta_cast, 4),
            temp=round(0.002 + 0.008 * max(cool_need, 0.0) - 0.014 * warm_excess, 4),
        )
        hsl = {
            "red": [-0.05, round((0.11 + 0.07 * max(chroma_headroom, 0.0)) * hsl_c, 3), -0.09],
            "orange": [0.02, round((-0.02 + 0.04 * max(chroma_headroom, 0.0)) * hsl_c, 3), round(0.04 + 0.03 * skin_present, 3)],
            "yellow": [-0.10, -0.08, -0.06],
            "green": [0.08, round(-0.13 + 0.05 * max(chroma_headroom, 0.0), 3), round(-0.10 + 0.05 * max(dark_mid, 0.0), 3)],
            "aqua": [0.0, -0.08, -0.06],
            "blue": [-0.05, round((0.07 + 0.07 * max(chroma_headroom, 0.0)) * hsl_c, 3), round(-0.11 - 0.06 * blue_present - 0.03 * hi_pressure, 3)],
        }
        split_s = 0.85 + 0.25 * max(chroma_headroom, -0.3)
        grade = dict(
            vibrance=round(0.07 + 0.09 * chroma_headroom - 0.05 * chroma_guard, 4),
            highlights=round(-0.08 - 0.18 * hi_pressure, 4),
            shadows=round(0.04 + 0.19 * sh_lift_need, 4),
            blacks=round(-0.05 - 0.09 * milky_black + 0.11 * sh_lift_need, 4),
            clarity=round(0.10 + 0.07 * max(flatness, 0.0), 4),
            dehaze=round(0.02 + 0.10 * veil_need, 4),
            sharpen=0.10,
            vignette=round(-0.11 * (1.0 - 0.55 * sh_lift_need), 3),
            skin_protect=round(0.22 + 0.25 * skin_present, 4),
            tone_curve={"rgb": [[0.0, 0.0], [0.22, s_low], [0.50, 0.494], [0.78, s_hi], [1.0, 1.0]]},
            hsl=hsl,
            cg_shadows=[218.0, round(0.09 * split_s, 3), round(-0.035 * (1.0 - 0.85 * sh_lift_need), 4)],
            cg_midtones=[35.0, round(0.025 * split_s, 3), 0.0],
            cg_highlights=[43.0, round(0.08 * split_s, 3), 0.02],
            cg_blending=0.68,
        )
    elif preset == "linear":
        # Wide-latitude, flat, neutral 16-bit grading canvas: zero crushed blacks, zero clipped highlights,
        # zero stylized HSL/split-tone/vignette/grain/sharpen so colorists have maximum headroom in Lr/PS.
        mix = 0.50
        offsets = dict(
            ev=round(0.08 * (1.0 - 0.65 * hi_pressure) + 0.08 * max(dark_mid, 0.0) * (1.0 - 0.65 * hi_pressure) - 0.06 * bright_mid, 4),
            contrast=round((-0.20 - 0.06 * max(-flatness, 0.0)) * float(np.clip((dr - 0.15) / 0.35, 0.30, 1.0)), 4),
            sat=round(-0.08 + 0.04 * max(chroma_headroom, 0.0) - 0.03 * chroma_guard, 4),
            shoulder=round(0.18 + 0.12 * hi_pressure, 4),
            toe=round(0.08 + 0.12 * sh_lift_need - 0.06 * milky_black, 4),
            tint=round(0.022 * green_cast - 0.020 * magenta_cast, 4),
        )
        grade = dict(
            highlights=round(-0.15 - 0.17 * hi_pressure, 4),
            shadows=round(0.12 + 0.14 * sh_lift_need, 4),
            blacks=round(0.04 * (1.0 - milky_black) - 0.02 * milky_black, 4),
        )
    elif preset == "frontier":
        mix = round(0.50 + 0.04 * (1.0 - chroma_guard) + 0.06 * max(chroma_headroom, 0.0), 3)
        ev_adj = round(0.06 * (1.0 - 0.60 * hi_pressure) + 0.10 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.07 * bright_mid, 4)
        hsl_c = 1.0 - 0.85 * chroma_guard
        offsets = dict(
            ev=ev_adj,
            contrast=round(0.06 + 0.09 * flatness - 0.02 * sh_lift_need, 4),
            sat=round(0.05 + 0.08 * chroma_headroom - 0.05 * chroma_guard, 4),
            shoulder=round(-0.01 + 0.12 * hi_pressure, 4),
            toe=round(-0.09 - 0.13 * milky_black + 0.15 * sh_lift_need, 4),
            tint=round(0.003 + 0.028 * green_cast - 0.022 * magenta_cast, 4),
            temp=round(0.001 + 0.009 * max(cool_need, 0.0) - 0.015 * warm_excess, 4),
        )
        hsl = {
            "green": [-0.08, round((0.08 + 0.05 * max(chroma_headroom, 0.0)) * hsl_c, 3), -0.03],
            "orange": [0.0, round((0.04 + 0.03 * max(chroma_headroom, 0.0)) * hsl_c, 3), round(0.03 + 0.03 * skin_present, 3)],
            "blue": [-0.03, round(0.04 * max(chroma_headroom, 0.0) * hsl_c, 3), round(-0.04 * blue_present - 0.03 * hi_pressure, 3)],
        }
        grade = dict(
            vibrance=round(0.09 + 0.09 * chroma_headroom - 0.06 * chroma_guard, 4),
            highlights=round(-0.05 - 0.19 * hi_pressure, 4),
            shadows=round(0.14 * sh_lift_need, 4),
            blacks=round(-0.03 - 0.08 * milky_black + 0.08 * sh_lift_need, 4),
            whites=round(0.04 * (1.0 - hi_pressure), 4),
            clarity=round(0.04 + 0.05 * max(flatness, 0.0), 4),
            dehaze=round(0.07 * veil_need, 4),
            skin_protect=round(0.18 * skin_present, 4),
            hsl=hsl,
        )
    elif preset == "noritsu":
        mix = round(0.50 + 0.03 * (1.0 - chroma_guard) + 0.04 * max(chroma_headroom, 0.0), 3)
        hsl_c = 1.0 - 0.85 * chroma_guard
        offsets = dict(
            ev=round(0.08 * (1.0 - 0.55 * hi_pressure) + 0.09 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.06 * bright_mid, 4),
            contrast=round(0.03 + 0.07 * flatness - 0.02 * sh_lift_need, 4),
            sat=round(0.04 + 0.06 * chroma_headroom - 0.04 * chroma_guard, 4),
            shoulder=round(0.05 + 0.11 * hi_pressure, 4),
            toe=round(-0.05 - 0.09 * milky_black + 0.10 * sh_lift_need, 4),
            tint=round(0.002 + 0.026 * green_cast - 0.022 * magenta_cast, 4),
            temp=round(0.002 + 0.008 * max(cool_need, 0.0) - 0.014 * warm_excess, 4),
        )
        grade = dict(
            vibrance=round(0.07 + 0.07 * chroma_headroom - 0.04 * chroma_guard, 4),
            highlights=round(-0.07 - 0.19 * hi_pressure, 4),
            shadows=round(0.05 + 0.14 * sh_lift_need, 4),
            blacks=round(-0.02 - 0.07 * milky_black + 0.05 * sh_lift_need, 4),
            skin_protect=round(0.20 + 0.26 * skin_present, 4),
            hsl={
                "yellow": [-0.04, -0.06, 0.02],
                "blue": [-0.04, round(0.04 * max(chroma_headroom, 0.0) * hsl_c, 3), round(-0.03 * blue_present, 3)],
            },
            cg_highlights=[46.0, 0.05, 0.01],
            cg_blending=0.72,
        )
    elif preset == "natural":
        offsets = dict(
            ev=round(0.07 * (1.0 - 0.60 * hi_pressure) + 0.08 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.06 * bright_mid, 4),
            contrast=round(-0.05 + 0.06 * flatness, 4),
            temp=round(-0.012 * warm_excess, 4),
            sat=round(-0.02 + 0.04 * max(chroma_headroom, 0.0) - 0.03 * chroma_guard, 4),
            shoulder=round(0.07 + 0.09 * hi_pressure, 4),
            tint=round(0.022 * green_cast - 0.018 * magenta_cast, 4),
        )
        grade = dict(
            shadows=round(0.06 + 0.13 * sh_lift_need, 4),
            highlights=round(-0.06 - 0.15 * hi_pressure, 4),
            blacks=round(-0.05 * milky_black + 0.03 * sh_lift_need, 4),
            vibrance=round(0.04 + 0.06 * max(chroma_headroom, 0.0) - 0.03 * chroma_guard, 4),
            cg_highlights=[42.0, 0.06, 0.01],
            cg_blending=0.72,
        )
    elif preset == "portrait":
        offsets = dict(
            ev=round(0.09 * (1.0 - 0.55 * hi_pressure) + 0.10 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.05 * bright_mid, 4),
            contrast=round(-0.08 + 0.05 * flatness, 4),
            sat=round(-0.03 + 0.04 * max(chroma_headroom, 0.0) - 0.03 * chroma_guard, 4),
            shoulder=round(0.09 + 0.10 * hi_pressure, 4),
            temp=round(0.005 + 0.009 * max(cool_need, 0.0) - 0.012 * warm_excess, 4),
            tint=round(0.004 + 0.024 * green_cast - 0.018 * magenta_cast, 4),
        )
        grade = dict(
            shadows=round(0.08 + 0.13 * sh_lift_need, 4),
            highlights=round(-0.09 - 0.17 * hi_pressure, 4),
            blacks=round(-0.05 * milky_black + 0.03 * sh_lift_need, 4),
            skin_protect=round(0.32 + 0.30 * skin_present, 4),
            vibrance=round(0.05 + 0.05 * max(chroma_headroom, 0.0) - 0.03 * chroma_guard, 4),
            hsl={
                "orange": [0.02, round(-0.05 * chroma_guard, 3), round(0.05 + 0.04 * skin_present, 3)],
                "yellow": [-0.08, -0.08, 0.03],
                "green": [0.04, -0.10, -0.02],
            },
        )
    elif preset == "vivid":
        mix = round(0.51 + 0.05 * (1.0 - chroma_guard) + 0.07 * max(chroma_headroom, 0.0), 3)
        hsl_c = 1.0 - 0.80 * chroma_guard
        offsets = dict(
            ev=round(0.05 * (1.0 - 0.60 * hi_pressure) + 0.09 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.07 * bright_mid, 4),
            contrast=round(0.09 + 0.10 * flatness - 0.03 * sh_lift_need, 4),
            sat=round(0.06 + 0.09 * chroma_headroom - 0.06 * chroma_guard, 4),
            shoulder=round(-0.02 + 0.12 * hi_pressure, 4),
            toe=round(-0.05 - 0.11 * milky_black + 0.12 * sh_lift_need, 4),
            tint=round(0.002 + 0.026 * green_cast - 0.022 * magenta_cast, 4),
        )
        grade = dict(
            vibrance=round(0.11 + 0.11 * chroma_headroom - 0.07 * chroma_guard, 4),
            highlights=round(-0.05 - 0.18 * hi_pressure, 4),
            shadows=round(0.13 * sh_lift_need, 4),
            blacks=round(-0.04 - 0.08 * milky_black + 0.09 * sh_lift_need, 4),
            clarity=round(0.06 + 0.06 * max(flatness, 0.0), 4),
            dehaze=round(0.09 * veil_need, 4),
            skin_protect=round(0.18 * skin_present, 4),
            hsl={
                "red": [-0.03, round((0.09 + 0.06 * max(chroma_headroom, 0.0)) * hsl_c, 3), -0.05],
                "blue": [-0.04, round((0.08 + 0.06 * max(chroma_headroom, 0.0)) * hsl_c, 3), round(-0.08 - 0.05 * blue_present, 3)],
            },
        )
    elif preset == "print_classic":
        s_amp = 0.026 + 0.016 * max(flatness, -0.4)
        offsets = dict(
            ev=round(0.06 * (1.0 - 0.60 * hi_pressure) + 0.09 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.06 * bright_mid, 4),
            contrast=round(0.08 + 0.08 * flatness - 0.03 * sh_lift_need, 4),
            sat=round(0.02 + 0.04 * chroma_headroom - 0.03 * chroma_guard, 4),
            shoulder=round(-0.03 + 0.13 * hi_pressure, 4),
            tint=round(0.002 + 0.024 * green_cast - 0.020 * magenta_cast, 4),
        )
        grade = dict(
            blacks=round(-0.05 - 0.08 * milky_black + 0.10 * sh_lift_need, 4),
            shadows=round(0.14 * sh_lift_need, 4),
            whites=round(0.04 * (1.0 - hi_pressure), 4),
            highlights=round(-0.05 - 0.17 * hi_pressure, 4),
            sharpen=0.10,
            tone_curve={
                "rgb": [
                    [0.0, 0.0],
                    [0.24, round(0.24 - s_amp * (1.0 - 0.72 * sh_lift_need), 4)],
                    [0.5, 0.50],
                    [0.76, round(0.76 + s_amp * (1.0 - 0.55 * hi_pressure), 4)],
                    [1.0, 1.0],
                ]
            },
            cg_highlights=[45.0, 0.07, 0.01],
            cg_blending=0.70,
        )
    elif preset == "cinema":
        split_scale = 0.85 + 0.30 * max(chroma_headroom, -0.3)
        offsets = dict(
            ev=round(0.05 * (1.0 - 0.60 * hi_pressure) + 0.09 * max(dark_mid, 0.0) * (1.0 - 0.60 * hi_pressure) - 0.07 * bright_mid, 4),
            contrast=round(0.03 + 0.06 * flatness, 4),
            sat=round(-0.02 + 0.04 * max(chroma_headroom, 0.0) - 0.03 * chroma_guard, 4),
            shoulder=round(0.11 + 0.09 * hi_pressure, 4),
            tint=round(0.022 * green_cast - 0.018 * magenta_cast, 4),
        )
        grade = dict(
            blacks=round(0.06 * (1.0 - milky_black) - 0.05 * milky_black + 0.04 * sh_lift_need, 4),
            highlights=round(-0.07 - 0.16 * hi_pressure, 4),
            shadows=round(0.10 * sh_lift_need, 4),
            grain=0.07,
            hsl={"green": [0.06, -0.12, -0.06]},
            cg_shadows=[198.0, round(0.14 * split_scale, 3), round(-0.03 * (1.0 - 0.80 * sh_lift_need), 4)],
            cg_highlights=[36.0, round(0.12 * split_scale, 3), 0.02],
            cg_blending=0.65,
            clarity=round(0.04 + 0.04 * max(flatness, 0.0), 4),
        )
    else:
        return dict(camera_matrix_mix=spec.get("camera_matrix_mix"), offsets=dict(spec["offsets"]), grade=dict(spec["grade"]))

    if film == "bw":
        mix = None
        for ck in ("vibrance", "skin_protect", "hsl", "cg_shadows", "cg_midtones", "cg_highlights"):
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
def _build_f32_lut3d(key: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (hi, sh, wh, bl, vib, sp, cgs, cgm, cgh, cgb, cgbl, hsl, tc, film) = key
    g = DarkroomGrade(
        highlights=hi, shadows=sh, whites=wh, blacks=bl, vibrance=vib, skin_protect=sp,
        cg_shadows=cgs, cg_midtones=cgm, cg_highlights=cgh, cg_balance=cgb, cg_blending=cgbl, hsl=dict(hsl),
    )
    lut = _apply_pointwise_exact(_LUT3D_HI_GRID, g, tc, film)
    return tuple(np.ascontiguousarray(lut[..., c]) for c in range(3))


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
    lut_c = _build_f32_lut3d(_pointwise_cache_key(g, tc, film))
    h, w = u.shape[:2]
    out = u if inplace else np.empty_like(u)
    scale = np.float32(_LUT3D_HI_N - 1)

    def _run(rr):
        r0, r1 = rr
        ch = u[r0:r1]
        n_px = (r1 - r0) * w
        coords = np.empty((3, n_px), dtype=np.float32)
        for c in range(3):
            np.clip(ch[..., c].ravel(), np.float32(0.0), np.float32(1.0), out=coords[c])
            coords[c] *= scale
        buf = np.empty(n_px, dtype=np.float32)
        dst = out[r0:r1]
        for c in range(3):
            map_coordinates(lut_c[c], coords, output=buf, order=1, mode="nearest", prefilter=False)
            np.clip(buf, np.float32(0.0), np.float32(1.0), out=buf)
            dst[..., c] = buf.reshape(r1 - r0, w)

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
    """Chunked multi-threaded luminance-only unsharp mask with overlap halo damping (~8-10x faster)."""
    h, w = e.shape[:2]
    sigma = max(0.8, min(h, w) * 0.0012)
    pad = max(3, int(math.ceil(2.5 * sigma)))
    coeff = np.float32(1.25 * sharpen)
    # Note: when inplace=True, read overlapping halo strips from e before mutating -> use out buffer only if small,
    # or precompute strip deltas? Overlapping reads from e would race if another thread mutates e[r0:r1].
    # So compute into out (or read-only e -> out) unless single-chunk; still 8x faster and tiny temp RAM!
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
        rng = np.random.default_rng(20261121 + r0 * 131)
        noise = rng.standard_normal((r1 - r0, w), dtype=np.float32)
        noise = uniform_filter(noise, size=3, mode="nearest")
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
    u = np.asarray(display, dtype=np.float32)
    owned = bool(inplace) or (u is not display)

    # 1. Spatial Dehaze first (if active)
    if g.dehaze:
        u = _apply_dehaze(u, g.dehaze, film=film, inplace=owned)
        owned = True

    # 2. Point-wise color/tone stack: zonal, vibrance, tone curve, HSL, skin protect, 3-way color grading
    tc = tuple((ch, pts) for ch, pts in g.tone_curve if film != "bw" or ch == "rgb") if g.tone_curve else ()
    has_1d = bool(g.highlights or g.shadows or g.whites or g.blacks or tc)
    has_3d = bool((g.vibrance and film != "bw") or (g.hsl and film != "bw")
                  or (g.skin_protect and film != "bw") or g.has_color_grading())
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
        u = _apply_sharpen(u, g.sharpen)
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
        out = np.ascontiguousarray(np.rot90(out, gm.rot90))
    if gm.flip_h:
        out = np.ascontiguousarray(out[:, ::-1])
    if gm.flip_v:
        out = np.ascontiguousarray(out[::-1, :])
    if abs(gm.angle) >= 1e-4:
        h, w = out.shape[:2]
        scale = _inscribed_scale(h, w, gm.angle)
        rad = math.radians(gm.angle)
        ca, sa = math.cos(rad) / scale, math.sin(rad) / scale
        if fast_preview:
            from PIL import Image
            # PIL affine uses (x, y) = (col, row) inverse mapping
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            # Note: (x_src, y_src) = R^T * (x - cx, y - cy) / scale + (cx, cy)
            # Matches scipy row/col convention: row' = ca*r - sa*c, col' = sa*r + ca*c
            a_m, b_m = ca, sa
            d_m, e_m = -sa, ca
            c_m = cx - a_m * cx - b_m * cy
            f_m = cy - d_m * cx - e_m * cy
            u8_f = np.clip(np.asarray(out, dtype=np.float32), np.float32(0.0), np.float32(1.0)); u8_f *= np.float32(255.0); u8_f += np.float32(0.5); u8 = u8_f.astype(np.uint8)
            im = Image.fromarray(u8, mode="RGB").transform(
                (w, h), Image.Transform.AFFINE, (a_m, b_m, c_m, d_m, e_m, f_m), resample=Image.Resampling.BILINEAR
            )
            out = np.asarray(im, dtype=np.float32) * np.float32(1.0 / 255.0)
        else:
            mat = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
            center = np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=np.float64)
            offset = center - mat @ center
            rot_out = np.empty_like(out)
            src_cs = [np.ascontiguousarray(out[..., c]) for c in range(out.shape[2])]
            def _rot_chunk(rr):
                r0, r1 = rr
                off_r = offset + mat[:, 0] * r0
                for c in range(len(src_cs)):
                    dst_c = np.empty((r1 - r0, w), dtype=np.float32)
                    affine_transform(src_cs[c], mat, offset=off_r, output_shape=(r1 - r0, w),
                                     output=dst_c, order=1, mode="nearest", prefilter=False)
                    np.clip(dst_c, np.float32(0.0), np.float32(1.0), out=dst_c)
                    rot_out[r0:r1, :, c] = dst_c
            _run_row_chunks(h, _rot_chunk, step=256)
            out = rot_out
    if apply_crop and gm.crop is not None:
        t, l, b, r = gm.crop
        h, w = out.shape[:2]
        y0 = max(0, min(h - 2, int(round(t * h))))
        x0 = max(0, min(w - 2, int(round(l * w))))
        y1 = max(y0 + 1, min(h, int(round(b * h))))
        x1 = max(x0 + 1, min(w, int(round(r * w))))
        out = np.ascontiguousarray(out[y0:y1, x0:x1])
    return out


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
    if not (fmt == "tiff16" and fast_tiff and (max_edge is None or int(max_edge) <= 0)):
        arr = np.clip(arr, np.float32(0.0), np.float32(1.0))
    h, w = arr.shape[:2]

    if max_edge is not None and int(max_edge) > 0 and max(h, w) > int(max_edge):
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
    icc = icc_profile("srgb")
    if fmt == "jpeg":
        q = int(np.clip(int(quality), 60, 100))
        im.save(path, format="JPEG", quality=q, subsampling=0 if q >= 92 else 2, optimize=True, icc_profile=icc)
    else:
        im.save(path, format="PNG", compress_level=6, icc_profile=icc)
    return path
