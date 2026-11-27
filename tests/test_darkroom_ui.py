"""Tests for the darkroom grading, geometry, adaptive presets, and export_image core."""
from __future__ import annotations

import numpy as np
import pytest

from filmcolor.darkroom import (
    DarkroomGrade,
    Geometry,
    T24_STYLE_PRESETS,
    analyze_scene_metrics,
    apply_darkroom_grade,
    apply_geometry,
    apply_vignette,
    resolve_adaptive_preset,
)


def test_darkroom_grade_and_geometry_core():
    rng = np.random.default_rng(7)
    img = rng.uniform(0.1, 0.9, (120, 160, 3)).astype(np.float32)

    # Identity is exact no-op
    assert apply_darkroom_grade(img, DarkroomGrade()) is img
    assert apply_geometry(img, Geometry()) is img

    # Full grade produces valid [0, 1] float32 in both fast and exact modes
    g = DarkroomGrade(
        highlights=-0.2, shadows=0.25, whites=0.1, blacks=-0.05, vibrance=0.2, skin_protect=0.4,
        tone_curve={"rgb": [[0.0, 0.02], [0.5, 0.52], [1.0, 0.98]]},
        cg_shadows=(210.0, 0.25, -0.05), cg_midtones=(30.0, 0.1, 0.02), cg_highlights=(42.0, 0.2, 0.04),
        hsl={"orange": (0.1, 0.15, 0.1), "blue": (-0.1, 0.2, -0.15)},
        dehaze=0.15, clarity=0.2, sharpen=0.35, vignette=-0.3, grain=0.15,
    )
    out_fast = apply_darkroom_grade(img, g, fast=True)
    out_exact = apply_darkroom_grade(img, g, fast=False)
    assert out_fast.shape == img.shape and np.all(np.isfinite(out_fast))
    assert out_exact.shape == img.shape and np.all(np.isfinite(out_exact))
    assert float(np.abs(out_fast - out_exact).mean()) < 0.02

    # B&W skips color-only ops (vibrance, hsl, skin_protect) while allowing tonal + split-tone
    gray = np.repeat(img[..., 1:2], 3, axis=-1)
    g_col = DarkroomGrade(vibrance=0.6, skin_protect=0.8, hsl={"orange": (0.4, 0.5, 0.3)})
    out_bw = apply_darkroom_grade(gray, g_col, film="bw", fast=False)
    assert np.allclose(out_bw[..., 0], out_bw[..., 1]) and np.allclose(out_bw[..., 1], out_bw[..., 2])

    # Geometry rot90 + flip + straighten + crop + vignette
    gm = Geometry(rot90=1, angle=4.0, flip_h=True, crop=(0.1, 0.1, 0.9, 0.9))
    g_out = apply_vignette(apply_geometry(img, gm, fast_preview=False), -0.3)
    assert g_out.shape == (int(round(0.8 * 160)), int(round(0.8 * 120)), 3)
    assert np.all(np.isfinite(g_out)) and g_out.min() >= 0.0 and g_out.max() <= 1.0



def test_adaptive_presets_respond_to_scene_metrics():
    import numpy as np
    from filmcolor.darkroom import analyze_scene_metrics, resolve_adaptive_preset

    # Flat, muted, hazy-black image vs high-contrast, vivid, clipping-highlight image
    flat_img = np.full((64, 64, 3), [0.38, 0.44, 0.37], dtype=np.float32)
    flat_img[:32, :, :] = [0.58, 0.64, 0.56]
    vivid_img = np.zeros((64, 64, 3), dtype=np.float32)
    vivid_img[:32, :, :] = [0.99, 0.96, 0.92]
    vivid_img[32:56, :, :] = [0.02, 0.25, 0.85]
    vivid_img[56:, :, :] = [0.01, 0.02, 0.04]

    m_flat = analyze_scene_metrics(flat_img)
    m_vivid = analyze_scene_metrics(vivid_img, above_white=0.15)

    # Standard must stay strictly unchanged
    std = resolve_adaptive_preset("standard", "color", m_flat)
    assert std["offsets"] == {} and std["grade"] == {}

    # Frontier should boost contrast/sat/vibrance more on flat+muted than on contrasty+vivid,
    # and pull highlights harder when highlight pressure is high
    f_flat = resolve_adaptive_preset("frontier", "color", m_flat)
    f_vivid = resolve_adaptive_preset("frontier", "color", m_vivid)
    assert f_flat["offsets"]["contrast"] > f_vivid["offsets"]["contrast"]
    assert f_flat["offsets"]["sat"] > f_vivid["offsets"]["sat"]
    assert f_flat["grade"]["vibrance"] > f_vivid["grade"]["vibrance"]
    assert f_vivid["grade"]["highlights"] < f_flat["grade"]["highlights"]
    assert f_flat["grade"]["blacks"] < f_vivid["grade"]["blacks"]  # anchors milky blacks

    # B&W strips color-only grading keys
    f_bw = resolve_adaptive_preset("frontier", "bw", m_flat)
    assert f_bw["camera_matrix_mix"] is None
    assert not any(k in f_bw["grade"] for k in ("vibrance", "skin_protect", "hsl"))



def test_linear_and_leica_presets():
    import numpy as np
    from filmcolor.darkroom import T24_STYLE_PRESETS, analyze_scene_metrics, resolve_adaptive_preset

    assert "linear" in T24_STYLE_PRESETS and "leica" in T24_STYLE_PRESETS
    img = np.linspace(0.05, 0.95, 64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)
    m = analyze_scene_metrics(img, above_white=0.08)

    lin = resolve_adaptive_preset("linear", "color", m)
    assert lin["offsets"]["contrast"] < -0.10
    assert lin["grade"]["highlights"] < -0.10
    assert lin["grade"]["shadows"] > 0.10
    assert not any(k in lin["grade"] for k in ("hsl", "cg_shadows", "cg_highlights", "vignette", "grain", "sharpen"))

    lei = resolve_adaptive_preset("leica", "color", m)
    assert lei["grade"]["clarity"] >= 0.08
    assert lei["grade"]["vignette"] < 0.0
    assert "red" in lei["grade"]["hsl"] and "blue" in lei["grade"]["hsl"]


def test_darkroom_audit_bugfixes_and_parity(tmp_path):
    """Regression tests for B01-B07 and P02/P08 in darkroom.py."""
    import numpy as np
    import pytest
    from PIL import Image
    from filmcolor.darkroom import (
        DarkroomGrade,
        Geometry,
        T24_STYLE_PRESETS,
        _apply_grain,
        _apply_pointwise_exact,
        _apply_pointwise_lut3d_f32,
        _apply_sharpen,
        analyze_scene_metrics,
        apply_darkroom_grade,
        apply_geometry,
        apply_lr_crop_angle,
        export_image,
        resolve_adaptive_preset,
    )
    from filmcolor.export import icc_profile

    # B01: apply_geometry fast_preview=False supports 2D grayscale and returns float32
    H, W = 120, 160
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ramp = np.stack([xx / W, yy / H, np.full((H, W), 0.5, dtype=np.float32)], axis=-1)
    g2d = apply_geometry(ramp[..., 0], Geometry(angle=5.0), fast_preview=False)
    assert g2d.shape == (H, W) and g2d.dtype == np.float32

    # B02: fast_preview=True vs SciPy affine parity (< 2/255 quantization error)
    g_fast = apply_geometry(ramp, Geometry(angle=10.0), fast_preview=True)
    g_full = apply_geometry(ramp, Geometry(angle=10.0), fast_preview=False)
    assert float(np.max(np.abs(g_fast[4:-4, 4:-4] - g_full[4:-4, 4:-4]))) < 0.01
    c_fast = apply_lr_crop_angle(ramp, (0.1, 0.1, 0.9, 0.9), 10.0, fast_preview=True)
    c_full = apply_lr_crop_angle(ramp, (0.1, 0.1, 0.9, 0.9), 10.0, fast_preview=False)
    assert float(np.max(np.abs(c_fast[4:-4, 4:-4] - c_full[4:-4, 4:-4]))) < 0.01

    # B03: export_image embeds requested ICC profile for PNG/JPEG
    png_lin = tmp_path / "out_lin.png"
    export_image(png_lin, ramp, fmt="png", encoding="linear")
    with Image.open(png_lin) as im:
        assert im.info.get("icc_profile") == icc_profile("linear")

    # B04: _apply_grain has no variance jump across 256-row chunk seam
    flat = np.full((512, 128, 3), 0.5, dtype=np.float32)
    g_out = _apply_grain(flat, 0.5)
    row_std = np.std(g_out[..., 0] - 0.5, axis=1)
    med_std = float(np.median(row_std[10:250]))
    assert abs(float(row_std[255]) - med_std) / med_std < 0.2
    assert abs(float(row_std[256]) - med_std) / med_std < 0.2

    # B05: resolve_adaptive_preset deep-copies globals & strips B&W keys when scene_metrics is None
    res_c = resolve_adaptive_preset("frontier", film="color", scene_metrics=None)
    assert res_c["grade"]["hsl"] is not T24_STYLE_PRESETS["frontier"]["grade"]["hsl"]
    res_bw = resolve_adaptive_preset("frontier", film="bw", scene_metrics=None)
    assert res_bw["camera_matrix_mix"] is None
    assert not any(k in res_bw["grade"] for k in ("hsl", "vibrance", "cg_shadows", "cg_blending", "cg_balance"))

    # B06: analyze_scene_metrics ignores NaN pixels and handles 2D/4-ch inputs
    nan_img = ramp.copy()
    nan_img[0, 0, 0] = np.nan
    m_nan = analyze_scene_metrics(nan_img)
    assert all(np.isfinite(v) for v in m_nan.values())
    m_2d = analyze_scene_metrics(ramp[..., 0])
    assert all(np.isfinite(v) for v in m_2d.values())
    m_4ch = analyze_scene_metrics(np.concatenate([ramp, np.ones((H, W, 1), dtype=np.float32)], axis=-1))
    assert all(np.isfinite(v) for v in m_4ch.values())

    # B07: apply_lr_crop_angle rejects non-finite angle and malformed crops
    with pytest.raises(ValueError, match="angle"):
        apply_lr_crop_angle(ramp, (0.1, 0.1, 0.9, 0.9), float("nan"))
    with pytest.raises(ValueError, match="crop"):
        apply_lr_crop_angle(ramp, (0.9, 0.1, 0.1, 0.9), 0.0)

    # P02: vectorized float32 trilinear LUT3D matches exact pointwise closely
    gr = DarkroomGrade(highlights=-0.2, shadows=0.25, vibrance=0.2, hsl={"orange": [0.1, 0.1, 0.0]})
    rng = np.random.default_rng(7)
    sample = rng.random((64, 64, 3), dtype=np.float32)
    lut_res = _apply_pointwise_lut3d_f32(sample, gr, (), "color")
    exact_res = _apply_pointwise_exact(sample, gr, (), "color")
    assert float(np.max(np.abs(lut_res - exact_res))) < 0.02

    # P08: _apply_sharpen inplace=True is bit-exact with inplace=False, and vignette-only no-op returns input
    sh_copy = sample.copy()
    assert np.array_equal(_apply_sharpen(sample, 0.4, inplace=False), _apply_sharpen(sh_copy, 0.4, inplace=True))
    vig_only = apply_darkroom_grade(sample, DarkroomGrade(vignette=-0.3), apply_vignette_here=False)
    assert vig_only is sample
