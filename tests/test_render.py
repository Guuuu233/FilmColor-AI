import json

import numpy as np
import pytest

from filmcolor.invert import InversionParams, srgb_oetf, tone_map
from filmcolor.render import (BOUNDS, LUMA, MID_GREY, PER_IMAGE, RenderParams, render, shoulder_curve,
                              srgb_eotf, wb_gains)


def scene(seed=0, n=4000):
    """Scene-linear test pixels: greys from deep shadow to 4x over white, plus saturated colours."""
    rng = np.random.default_rng(seed)
    grey = np.geomspace(1e-3, 4, n // 2)[:, None].repeat(3, 1)
    colour = rng.uniform(0, 1.5, (n // 2, 3)) * rng.uniform(0.2, 1, (n // 2, 1))
    return np.concatenate([grey, colour]).astype(np.float32)[None]


def ramp(n=1000, hi=4.0):
    return np.linspace(0, hi, n, dtype=np.float32)[None, :, None].repeat(3, -1)


def is_grey(img, atol=1e-6):
    return np.allclose(img[..., 0], img[..., 1], atol=atol) and np.allclose(img[..., 2], img[..., 1], atol=atol)


def test_defaults_reproduce_sprint1_display_exactly():
    P = scene() - 0.01  # includes values below black
    out, diag = render(P, RenderParams())
    ref = tone_map(P, InversionParams(base=(0.5, 0.4, 0.1)))
    assert out.dtype == np.float32 and np.array_equal(out, ref)
    assert diag["below_zero_after_colour"] > 0 and diag["above_white"] > 0


@pytest.mark.parametrize("kw", [dict(), dict(ev=1.3), dict(sat=0.3), dict(sat=1.8), dict(contrast=1.7),
                                dict(midtone=0.3), dict(shoulder=0.6),
                                dict(matrix=((1.3, -0.2, -0.1), (-0.1, 1.2, -0.1), (0.0, -0.3, 1.3)))])
def test_neutral_inputs_stay_neutral_without_white_balance(kw):
    out, _ = render(ramp(), RenderParams(**kw))
    assert is_grey(out)


def test_white_balance_tints_neutrals_but_keeps_their_luma():
    g = wb_gains(0.5, -0.3)
    assert LUMA @ g == pytest.approx(1)
    assert g[0] > g[2]                       # warm: red up, blue down
    assert wb_gains(0, 0.6)[1] < 1           # magenta: green down
    out, _ = render(ramp(hi=0.9), RenderParams(temp=0.5))
    assert out[0, -1, 0] > out[0, -1, 1] > out[0, -1, 2]


def test_saturation_zero_gives_luma_grey():
    P = scene()
    out, _ = render(P, RenderParams(sat=0))
    assert is_grey(out, atol=1e-5)


@pytest.mark.parametrize("seed", range(20))
def test_tone_curve_monotone_with_fixed_endpoints_for_random_params(seed):
    rng = np.random.default_rng(seed)
    p = RenderParams(**{k: rng.uniform(*BOUNDS[k]) for k in ("ev", "contrast", "midtone", "shoulder")})
    out, _ = render(ramp(4000, 8.0), p)
    y = out[0, :, 1]
    assert np.all(np.diff(y) >= -1e-7)
    assert y[0] == 0 and y.max() <= 1


def test_shoulder_maps_its_white_to_one_and_is_identity_when_zero():
    x = np.array([0.0, 0.18, 1.0, 3.0], np.float32)
    np.testing.assert_array_equal(shoulder_curve(x, 0), x)
    assert shoulder_curve(np.float32(1 + 7 * 0.4), 0.4) == pytest.approx(1, abs=1e-6)
    assert np.all(np.diff(shoulder_curve(np.linspace(0, 20, 500, dtype=np.float32), 1.0)) > 0)


def test_midtone_places_scene_grey_and_keeps_white():
    for shoulder in (0.0, 0.5):
        p = RenderParams(midtone=0.3, shoulder=shoulder)
        out, _ = render(np.array([[[MID_GREY] * 3, [0.0] * 3]], np.float32), p)
        assert srgb_eotf(out[0, 0, 0]) == pytest.approx(0.3, abs=1e-5)
        assert out[0, 1, 0] == 0
    out, _ = render(np.array([[[1.0] * 3]], np.float32), RenderParams(midtone=0.3))
    assert out[0, 0, 0] == pytest.approx(1, abs=1e-6)


def test_contrast_keeps_pivot_and_endpoints_and_sets_slope():
    p = RenderParams(contrast=1.6, midtone=0.2)
    flat = RenderParams(midtone=0.2)
    pts = np.array([[[0.0] * 3, [MID_GREY] * 3, [1.0] * 3]], np.float32)
    out, _ = render(pts, p)
    assert out[0, 0, 0] == 0 and out[0, 2, 0] == pytest.approx(1, abs=1e-6)
    assert out[0, 1, 0] == pytest.approx(float(srgb_oetf(0.2)), abs=1e-6)  # scene grey -> pivot, unchanged
    near = np.array([[[MID_GREY * 0.999] * 3, [MID_GREY * 1.001] * 3]], np.float32).astype(np.float64)
    e1, _ = render(near, p)
    e0, _ = render(near, flat)
    slope = (float(e1[0, 1, 0]) - float(e1[0, 0, 0])) / (float(e0[0, 1, 0]) - float(e0[0, 0, 0]))
    assert slope == pytest.approx(1.6, rel=0.01)


def test_exposure_scales_scene_linear_input():
    P = scene()
    a, _ = render(P, RenderParams(ev=1.0))
    b, _ = render(P * 2, RenderParams())
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_matrix_and_saturation_negatives_are_clipped_and_reported():
    P = np.array([[[0.9, 0.05, 0.05]]], np.float32)  # saturated red
    out, diag = render(P, RenderParams(sat=2.0))
    assert diag["below_zero_after_colour"] > 0
    assert out.min() >= 0


@pytest.mark.parametrize("kw", [dict(ev=3.5), dict(temp=-1.1), dict(sat=-0.1), dict(contrast=0.4),
                                dict(midtone=0.6), dict(shoulder=1.2), dict(ev=float("nan")),
                                dict(matrix=((1.2, 0, 0), (0, 1, 0), (0, 0, 1))),            # row sum != 1
                                dict(matrix=((2.5, -1.0, -0.5), (0, 1, 0), (0, 0, 1))),      # diagonal too big
                                dict(matrix=((1.0, 0.9, -0.9), (0, 1, 0), (0, 0, 1))),       # off-diagonal too big
                                dict(matrix=((0.5, 0.5, 0.0), (0.5, 0.5, 0.0), (0, 0, 1))),  # singular
                                dict(matrix=((1, 0), (0, 1)))])
def test_invalid_parameters_are_rejected(kw):
    with pytest.raises(ValueError):
        RenderParams(**kw)


def test_extreme_parameters_give_finite_bounded_output():
    P = scene()
    for corner in range(2 ** len(PER_IMAGE)):
        kw = {k: BOUNDS[k][(corner >> i) & 1] for i, k in enumerate(PER_IMAGE)}
        out, _ = render(P, RenderParams(**kw))
        assert np.all(np.isfinite(out)) and out.min() >= 0 and out.max() <= 1, kw


def test_render_rejects_non_finite_input():
    with pytest.raises(ValueError):
        render(np.array([[[np.nan, 0, 0]]], np.float32), RenderParams())


def test_serialisation_roundtrip_and_determinism():
    p = RenderParams(ev=0.4, temp=-0.2, tint=0.1, sat=1.3, contrast=1.2, midtone=0.22, shoulder=0.3,
                     matrix=((1.2, -0.1, -0.1), (-0.05, 1.1, -0.05), (0.0, -0.2, 1.2)))
    q = RenderParams.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q == p
    P = scene()
    assert np.array_equal(render(P, p)[0], render(P, q)[0])
    with pytest.raises(ValueError):
        RenderParams.from_dict({"exposure": 1})


def _jacobian_conditioning(names, x0, P):
    """min/max singular value of the column-normalised finite-difference Jacobian."""
    base, _ = render(P, RenderParams(**x0))
    cols = []
    for k in names:
        h = 1e-2 * (BOUNDS[k][1] - BOUNDS[k][0])
        out, _ = render(P, RenderParams(**(x0 | {k: x0[k] + h})))
        cols.append(((out - base) / h).ravel())
    J = np.stack(cols, 1).astype(np.float64)
    J /= np.linalg.norm(J, axis=0)
    s = np.linalg.svd(J, compute_uv=False)
    return s.min() / s.max()


X0 = dict(ev=0.2, temp=0.1, tint=-0.1, sat=1.2, contrast=1.3, midtone=0.2, shoulder=0.4)


def test_per_image_parameters_are_identifiable():
    """No parameter (or combination) duplicates another: the Jacobian has full rank. ev and midtone
    are strongly correlated (|cos| ~ 0.9, both brighten midtones), so the 7-parameter set is only
    weakly conditioned (~0.03); see docs/SPRINT2_PARAMS.md."""
    assert _jacobian_conditioning(PER_IMAGE, X0, scene(n=2000)) > 0.02


def test_dropping_ev_leaves_a_well_conditioned_set():
    names = [k for k in PER_IMAGE if k != "ev"]
    assert _jacobian_conditioning(names, X0, scene(n=2000)) > 0.2


def test_diagnostics_respect_mask():
    P = np.array([[[0.5, 0.5, 0.5], [0.9, 0.0, 0.0], [3.0, 3.0, 3.0]]], np.float32)
    _, all_px = render(P, RenderParams(sat=2.0))
    _, img_px = render(P, RenderParams(sat=2.0), mask=np.array([[True, False, False]]))
    assert all_px["below_zero_after_colour"] > 0 and all_px["above_white"] > 0
    assert img_px["below_zero_after_colour"] == 0 and img_px["above_white"] == 0


def test_extended_darkroom_controls_and_tone_curve():
    """T4 (2026-10-23): highlights, shadows, whites, blacks, vibrance, slope_r/b and tone_curve are
    deterministic, serialisable, monotone on grey ramps, and preserve exact Sprint 1 defaults when zero."""
    from filmcolor.render import CURVE_PRESETS

    r = ramp(500, 1.5)
    base, _ = render(r, RenderParams())

    # Shadows lift darks more than highlights; highlights pull down brights more than darks
    sh_up, _ = render(r, RenderParams(shadows=0.5))
    hl_dn, _ = render(r, RenderParams(highlights=-0.6))
    assert float(sh_up[0, 50, 1] - base[0, 50, 1]) > float(sh_up[0, 300, 1] - base[0, 300, 1]) > 0
    assert float(hl_dn[0, 300, 1] - base[0, 300, 1]) < float(hl_dn[0, 50, 1] - base[0, 50, 1]) <= 0

    # Positive blacks lift pure black > 0 (paper-black margin); whites shift upper tones
    bk_up, _ = render(r, RenderParams(blacks=0.4))
    assert float(bk_up[0, 0, 1]) > 0.02 and is_grey(bk_up)

    # Vibrance boosts low-chroma pixels more than already-saturated pixels
    low_c = np.array([[[0.30, 0.28, 0.26]]], np.float32)
    high_c = np.array([[[0.55, 0.20, 0.10]]], np.float32)
    l0, _ = render(low_c, RenderParams())
    l1, _ = render(low_c, RenderParams(vibrance=0.8))
    h0, _ = render(high_c, RenderParams())
    h1, _ = render(high_c, RenderParams(vibrance=0.8))
    gain_low = float((l1.max() - l1.min()) / (l0.max() - l0.min()))
    gain_high = float((h1.max() - h1.min()) / (h0.max() - h0.min()))
    assert gain_low > gain_high >= 1.0

    # Per-channel slope ratios (slope_r, slope_b) pivot at MID_GREY
    piv = np.array([[[MID_GREY] * 3]], np.float32)
    sr, _ = render(piv, RenderParams(slope_r=0.3, slope_b=-0.2))
    assert is_grey(sr, atol=1e-5)

    # Monotone spline tone curve and presets round-trip through JSON
    p_tc = RenderParams(
        ev=0.1, highlights=-0.3, shadows=0.25, whites=0.1, blacks=0.15, vibrance=0.2,
        slope_r=0.1, slope_b=-0.1,
        tone_curve={"rgb": CURVE_PRESETS["standard"]["rgb"], "b": [[0.0, 0.02], [0.5, 0.52], [1.0, 0.98]]},
    )
    d = p_tc.to_dict()
    assert "highlights" in d and "tone_curve" in d
    p_rt = RenderParams.from_dict(json.loads(json.dumps(d)))
    assert p_rt == p_tc
    P = scene()
    assert np.array_equal(render(P, p_tc)[0], render(P, p_rt)[0])

    for bad in (dict(highlights=1.5), dict(slope_r=0.8), dict(tone_curve={"rgb": [[0.5, 0.2], [0.5, 0.8]]})):
        with pytest.raises(ValueError):
            RenderParams(**bad)


def test_t4c_chroma_preserve_keeps_midtone_ratios_and_leaves_deep_shadows():
    """T4c: with a midtone lift the per-channel power compresses R:G:B ratios (by the exponent g); chroma_preserve=1
    applies the curve to luminance instead, so a mid-tone pixel keeps its display-linear ratios, while deep shadows
    (weight 0) render as with chroma_preserve=0. chroma_preserve=0 is the previous renderer exactly."""
    from filmcolor.render import render_full, srgb_eotf

    x = np.array([[[0.06, 0.04, 0.025], [0.0004, 0.0003, 0.0005]]], np.float32)
    p0 = RenderParams(midtone=0.34, shoulder=0.4)
    e0, _, _ = render_full(x, p0)
    e1, _, _ = render_full(x, RenderParams(midtone=0.34, shoulder=0.4, chroma_preserve=1.0))
    lin0, lin1 = srgb_eotf(e0[0, 0]), srgb_eotf(e1[0, 0])
    assert lin1[0] / lin1[1] == pytest.approx(0.06 / 0.04, rel=0.01)       # ratios kept
    assert lin0[0] / lin0[1] < 0.06 / 0.04 * 0.95                        # compressed per channel
    assert np.array_equal(e0[0, 1], e1[0, 1])                            # deep shadow untouched
    assert RenderParams(chroma_preserve=0.0).to_dict() == RenderParams().to_dict()  # old sidecars unchanged
    with pytest.raises(ValueError):
        RenderParams(chroma_preserve=1.5)


def test_t4c_camera_color_matrix():
    """T4c: camera -> sRGB from LibRaw's XYZ -> camera matrix (dcraw construction), rows summing to 1, blended with
    the identity by ``mix``; files without a matrix raise."""
    from filmcolor.render import IDENTITY, XYZ_FROM_SRGB_D65, camera_color_matrix

    rgb_cam = np.array([[1.27, -0.09, -0.18], [-0.11, 1.42, -0.31], [0.0, -0.35, 1.35]])
    cam_rgb = np.linalg.inv(rgb_cam / rgb_cam.sum(1, keepdims=True))
    cam_xyz = cam_rgb @ np.linalg.inv(XYZ_FROM_SRGB_D65)  # a camera whose camera -> sRGB is rgb_cam
    full = np.array(camera_color_matrix(cam_xyz, 1.0))
    assert np.allclose(full, rgb_cam / rgb_cam.sum(1, keepdims=True), atol=1e-9)
    half = np.array(camera_color_matrix(cam_xyz, 0.5))
    assert np.allclose(half.sum(1), 1.0, atol=1e-12)
    assert np.allclose(half, 0.5 * full + 0.5 * np.eye(3), atol=1e-9)
    assert camera_color_matrix(cam_xyz, 0.0) == IDENTITY
    with pytest.raises(ValueError):
        camera_color_matrix(np.zeros((3, 3)), 0.5)


def test_t4y_camera_color_matrix_accepts_canon_eos_and_sony_a7_primaries():
    """T4y: MATRIX_DIAG=(0.5, 2.4), MATRIX_OFF_DIAG=(-1.6, 0.8) allows physical camera matrices
    from Canon EOS and Sony A7/A7II, while still rejecting pathological out-of-bound matrices."""
    from filmcolor.render import RenderParams, camera_color_matrix

    # Canon EOS 500D / 5D Mark II / 6D Mark II rgb_xyz_matrix
    canon_xyz = np.array([
        [0.6875, -0.0970, -0.0932],
        [-0.4691, 1.2459, 0.2501],
        [-0.0874, 0.1953, 0.5809],
    ])
    canon_full = camera_color_matrix(canon_xyz, 1.0)
    canon_mix = camera_color_matrix(canon_xyz, 0.4)
    assert np.allclose(np.sum(canon_full, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.sum(canon_mix, axis=1), 1.0, atol=1e-9)
    assert np.isclose(canon_full[0][0], 1.8622692, atol=1e-4)
    assert np.isclose(canon_mix[0][0], 1.3449077, atol=1e-4)
    # Validates cleanly through RenderParams(matrix=...)
    p_canon = RenderParams(matrix=canon_full)
    assert p_canon.matrix == canon_full

    # Sony ILCE-7 (A7) rgb_xyz_matrix
    sony_xyz = np.array([
        [0.5271, -0.0712, -0.0347],
        [-0.6153, 1.3653, 0.2763],
        [-0.1601, 0.2366, 0.7242],
    ])
    sony_full = camera_color_matrix(sony_xyz, 1.0)
    sony_mix = camera_color_matrix(sony_xyz, 0.4)
    assert np.allclose(np.sum(sony_full, axis=1), 1.0, atol=1e-9)
    assert np.allclose(np.sum(sony_mix, axis=1), 1.0, atol=1e-9)
    assert np.isclose(sony_full[0][0], 2.0123623, atol=1e-4)
    assert np.isclose(sony_mix[0][0], 1.4049449, atol=1e-4)
    p_sony = RenderParams(matrix=sony_full)
    assert p_sony.matrix == sony_full

    # Pathological matrices remain rejected
    with pytest.raises(ValueError, match="colour matrix diagonal outside"):
        RenderParams(matrix=((2.5, -1.0, -0.5), (-0.5, 2.0, -0.5), (-0.5, -0.5, 2.0)))
    with pytest.raises(ValueError, match="colour matrix off-diagonal outside"):
        RenderParams(matrix=((1.0, 0.9, -0.9), (-0.5, 2.0, -0.5), (-0.5, -0.5, 2.0)))
