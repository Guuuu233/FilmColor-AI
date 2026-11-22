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
