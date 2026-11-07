import json

import numpy as np
import pytest

from filmcolor.invert import (InversionParams, base_from_regions, density, estimate_white_point,
                              invert, invert_negative, range_report, srgb_oetf)

BASE = np.array([0.6, 0.45, 0.15], np.float32)       # orange mask
GAMMAS = np.array([0.55, 0.6, 0.7], np.float32)      # per-channel film gamma
E0, EMAX = 1e-3, 1.0                                 # exposure at base (toe) and at the white point


def negative_of(E):
    """Ideal negative scanned with no crosstalk: D = gamma_c * log10(E / E0) above base, 0 below."""
    D = GAMMAS * np.log10(np.maximum(E, E0) / E0)
    return BASE * 10 ** (-D)


def grey_ramp(n=256):
    E = np.geomspace(E0, EMAX, n, dtype=np.float32)
    return np.repeat(E[None, :, None], 3, -1)  # (1, n, 3), neutral


def known_params(gamma=0.6):
    d_hi = tuple(float(x) for x in GAMMAS * np.log10(EMAX / E0))
    return InversionParams(base=tuple(BASE), d_hi=d_hi, gamma=gamma)


def test_density_zero_at_base_and_log10_elsewhere():
    rgb = np.array([[BASE, BASE * 0.1, BASE * 2]], np.float32)
    D = density(rgb, BASE)
    np.testing.assert_allclose(D[0, 0], 0, atol=1e-6)
    np.testing.assert_allclose(D[0, 1], 1, atol=1e-6)
    np.testing.assert_allclose(D[0, 2], -np.log10(2), atol=1e-6)  # brighter than base -> negative


def test_density_finite_for_zero_and_negative_input():
    D = density(np.array([[[0, -0.01, 1e-9]]], np.float32), BASE)
    assert np.all(np.isfinite(D))


def test_grey_ramp_stays_neutral_despite_channel_gammas_and_mask():
    E = grey_ramp()
    P = invert(density(negative_of(E), BASE), known_params())
    np.testing.assert_allclose(P[..., 0], P[..., 1], atol=1e-5)
    np.testing.assert_allclose(P[..., 2], P[..., 1], atol=1e-5)
    assert np.all(np.diff(P[0, :, 1]) > 0)  # monotone


def test_recovers_scene_exposure_when_gamma_matches_green():
    E = grey_ramp()
    P = invert(density(negative_of(E), BASE), known_params(gamma=float(GAMMAS[1])))
    expected = (E / EMAX - E0 / EMAX) / (1 - E0 / EMAX)
    np.testing.assert_allclose(P, expected, atol=1e-5)


def test_black_and_white_points_map_to_0_and_1():
    p = known_params()
    D = density(negative_of(np.array([[[E0] * 3, [EMAX] * 3]], np.float32)), BASE)
    np.testing.assert_allclose(invert(D, p), [[[0, 0, 0], [1, 1, 1]]], atol=1e-5)


def test_white_point_percentile_ignores_masked_pixels():
    E = grey_ramp()
    D = density(negative_of(E), BASE)
    D_bad = D.copy()
    D_bad[0, :8] = 5.0  # e.g. a dense defect outside the frame
    mask = np.ones(D.shape[:2], bool)
    mask[0, :8] = False
    np.testing.assert_allclose(estimate_white_point(D_bad, mask, 100), estimate_white_point(D, None, 100), atol=1e-6)


def test_auto_white_point_balances_a_colour_cast_in_highlights():
    E = grey_ramp()
    D = density(negative_of(E), BASE)
    res = invert_negative(BASE * 10 ** (-D), BASE, white_pct=100)
    np.testing.assert_allclose(res.positive[0, -1], [1, 1, 1], atol=1e-4)
    np.testing.assert_allclose(res.positive[..., 0], res.positive[..., 1], atol=1e-4)


def test_range_report_counts_clipping():
    E = grey_ramp(1000)
    rgb = negative_of(E)
    res = invert_negative(rgb, BASE, white_pct=99.0)
    r = res.report["range"]
    for c in "RGB":
        assert 0.005 < r[c]["clip_high"] <= 0.011  # ~1% above the 99th percentile
        assert r[c]["below_black"] == 0
    rgb[0, :10] = BASE * 1.5  # light that did not pass through film
    r2 = invert_negative(rgb, BASE, d_hi=res.params.d_hi).report["range"]
    assert all(abs(r2[c]["brighter_than_base"] - 0.01) < 1e-9 for c in "RGB")
    assert all(r2[c]["below_black"] > 0 for c in "RGB")


def test_tone_map_is_srgb_encoded_and_bounded():
    np.testing.assert_allclose(srgb_oetf(np.array([0.0, 0.0031308, 0.18, 1.0, 2.0])),
                               [0, 0.04045, 0.4613, 1, 1], atol=1e-4)
    res = invert_negative(negative_of(grey_ramp()), BASE, exposure=2.0)
    assert res.display.min() >= 0 and res.display.max() <= 1


def test_params_validation_and_serialisation():
    with pytest.raises(ValueError):
        InversionParams(base=(0.5, 0, 0.1))
    with pytest.raises(ValueError):
        InversionParams(base=tuple(BASE), d_lo=(1, 1, 1), d_hi=(1, 2, 2))
    with pytest.raises(ValueError):
        InversionParams(base=tuple(BASE), gamma=0)
    d = known_params().to_dict()
    assert d["log_range"] == pytest.approx(0.6 * 3 / 0.6)
    json.dumps(d)


def test_base_from_regions_median_and_validation():
    rgb = np.empty((100, 100, 3), np.float32)
    rgb[:] = BASE
    rgb[60:, 60:] = 0.01
    b = base_from_regions(rgb, [(0, 0, 50, 50), (0, 50, 50, 50)])
    np.testing.assert_allclose(b["rgb"], BASE, rtol=1e-6)
    assert b["n"] == 5000
    with pytest.raises(ValueError):
        base_from_regions(rgb, [])
    with pytest.raises(ValueError):
        base_from_regions(rgb, [(90, 90, 20, 20)])


def test_invert_negative_is_deterministic():
    rgb = negative_of(grey_ramp())
    a, b = invert_negative(rgb, BASE), invert_negative(rgb, BASE)
    assert np.array_equal(a.display, b.display) and a.params == b.params


def test_block_white_point_ignores_single_pixel_noise_in_the_densest_channel():
    """T4b: a dense, nearly uniform blue sky a few ADU above black has a single-pixel high-density tail from noise;
    the block-mean white point stays at the scene's density, the default single-pixel one does not."""
    from filmcolor.invert import block_white_point

    rng = np.random.default_rng(3)
    h, w = 512, 512
    D = np.empty((h, w, 3), np.float32)
    D[...] = rng.uniform(0.1, 1.0, (h, w, 1)) * np.array([0.8, 1.1, 1.3], np.float32)
    D[: h // 4] = [0.8, 1.1, 1.3]                     # sky: the white point in every channel
    T = BASE * 10.0 ** -D
    T[..., 2] = np.maximum(T[..., 2] + rng.normal(0, 0.25 * T[..., 2].min(), (h, w)), 1e-5)  # blue noise
    mask = np.ones((h, w), bool)
    pix = invert_negative(T, BASE, mask=mask)
    blk = invert_negative(T, BASE, mask=mask, white_block=16, white_pct=99.8)
    assert pix.params.d_hi[2] > 1.3 + 0.2             # single pixels: pushed up by noise
    assert abs(blk.params.d_hi[2] - 1.3) < 0.03       # block means: the sky's density (residual noise / 16)
    assert abs(blk.params.d_hi[1] - 1.1) < 0.001
    assert blk.report["white_point"] == dict(method="block_mean", block=16, pct=99.8, n_blocks=(h // 16) * (w // 16))
    assert "white_point" not in pix.report            # default reports keep their field set

    # blocks with any pixel outside the mask are not used; too few blocks fall back to single pixels
    m2 = mask.copy()
    m2[::16] = False
    d_hi, info = block_white_point(T, BASE, m2, 16, 99.8)
    assert info["method"] == "pixel_fallback" and info["n_blocks"] == 0
    assert d_hi == estimate_white_point(density(T[m2], BASE), None, 99.8)


def test_t4c_neutral_ratio_white_point_moves_only_a_blue_sky_white_point():
    """T4c: a white point with the blue-sky signature (R/G below, B/G above the roll's neutral ratio) is moved 70 %
    towards the neutral ratio, its level set from the frame (0.94 x max_c d_hi_c / r_c); other white points stay."""
    from filmcolor.invert import neutral_ratio_white_point

    sky = (0.65, 1.0, 1.25)
    d, info = neutral_ratio_white_point(sky, (0.72, 1.10))
    assert info["applied"]
    r_eff = 0.3 * np.array([0.65, 1.0, 1.25]) + 0.7 * np.array([0.72, 1.0, 1.10])
    assert np.allclose(np.array(d) / d[1], r_eff)
    assert d[1] == pytest.approx(0.94 * max(sky / r_eff))
    neutral = (0.73, 1.0, 1.12)
    d_n, info_n = neutral_ratio_white_point(neutral, (0.72, 1.10))
    assert d_n == neutral and not info_n["applied"]
    rgb = BASE * 10.0 ** -np.array([[[0.1, 0.2, 0.3], [0.65, 1.0, 1.25]]], np.float32)
    res = invert_negative(rgb, BASE, white_pct=100, white_ratio=(0.72, 1.10))
    assert res.report["white_point"]["neutral_ratio"]["applied"]


def test_t4k_blue_white_guard_lifts_a_collapsed_blue_white_point():
    """T4k (DSC_3827: warm ridge as the densest area, d_hi_B / d_hi_G 1.02 against about 1.35 on its roll): below
    min_ratio the blue white point moves ``weight`` of the way to target_ratio x d_hi_G; other channels and a white
    point at or above min_ratio stay; invert_negative applies it only when given and records it."""
    from filmcolor.invert import blue_white_guard

    d, info = blue_white_guard((0.8, 1.0, 1.02), (1.12, 1.18, 0.65))
    assert info["applied"] and d[:2] == (0.8, 1.0)
    assert d[2] == pytest.approx(1.02 + 0.65 * (1.18 - 1.02)) and info["ratio_bg_used"] == pytest.approx(d[2])
    d_n, info_n = blue_white_guard((0.8, 1.0, 1.30), (1.12, 1.18, 0.65))
    assert d_n == (0.8, 1.0, 1.30) and not info_n["applied"]
    rgb = BASE * 10.0 ** -np.array([[[0.1, 0.2, 0.3], [0.8, 1.0, 1.02]]], np.float32)
    plain = invert_negative(rgb, BASE, white_pct=100)
    res = invert_negative(rgb, BASE, white_pct=100, white_b_guard=(1.12, 1.18, 0.65))
    assert plain.params.d_hi == pytest.approx((0.8, 1.0, 1.02), abs=1e-5) and "white_point" not in plain.report
    assert res.params.d_hi[2] == pytest.approx(d[2], abs=1e-4) and res.report["white_point"]["blue_guard"]["applied"]
    given = invert_negative(rgb, BASE, d_hi=(0.8, 1.0, 1.02), white_b_guard=(1.12, 1.18, 0.65))
    assert given.params.d_hi == (0.8, 1.0, 1.02)  # a given (replayed) white point is never changed


def test_t4m_blue_white_guard_is_gated_by_the_orange_base_and_replayed_once():
    from filmcolor.invert import blue_white_guard

    guard = (1.06, 1.18, 0.65, 0.48)
    d = (0.8, 1.0, 1.02)
    base = np.array((0.8, 0.5, 0.2), np.float32)
    lifted, info = blue_white_guard(d, guard, base=base)
    assert info["applied"] and info["base_bg"] == pytest.approx(0.4)
    assert lifted == pytest.approx((0.8, 1.0, 1.124))
    for bg in (0.48, 0.65, 1.05):
        unchanged, diag = blue_white_guard(d, guard, base=(0.8, 1.0, bg))
        assert unchanged == d and not diag["applied"]
    unchanged, diag = blue_white_guard((0.8, 1.0, 1.06), guard, base=base)
    assert unchanged[2] == 1.06 and not diag["applied"]
    assert blue_white_guard(d, guard[:3], base=(1, 1, 1))[1]["applied"]  # legacy three-value guard
    rgb = base * 10.0 ** -np.array([[[0.1, 0.2, 0.3], list(d)]], np.float32)
    result = invert_negative(rgb, base, white_pct=100, white_b_guard=guard)
    assert result.report["white_point"]["blue_guard"]["applied"]
    assert result.params.d_hi == pytest.approx(lifted, abs=1e-5)
    replay = invert_negative(rgb, base, d_hi=result.params.d_hi, white_b_guard=guard)
    np.testing.assert_array_equal(result.positive, replay.positive)
    assert "white_point" not in replay.report


def test_t4l_spike_clamp_holds_single_channel_spikes_and_keeps_the_holder():
    """T4l (A1106: blue at or below the black level behind the orange mask -> density floor 6 D -> positive ~36 000 x
    white): where the middle channel's normalised density is at most mid_max, one channel is held to max(middle, 1) +
    margin; an opaque holder (two channels far past white) and ordinary pixels are unchanged; the clamp is recorded in
    the parameters (replayed) and the report counts the pixels."""
    from filmcolor.invert import clamp_spikes

    s = np.array([[0.5, 0.6, 0.55],     # ordinary
                  [0.9, 1.0, 5.4],      # blue spike inside the image
                  [3.0, 2.5, 9.0],      # opaque holder: middle channel far past white
                  [1.02, 1.04, 1.10]],  # a highlight a little past white, within the margin
                 np.float32)
    out, n = clamp_spikes(s, (1.15, 0.12))
    assert n == 1 and out[1, 2] == pytest.approx(1.12)
    np.testing.assert_array_equal(out[[0, 2, 3]], s[[0, 2, 3]])
    rgb = BASE * 10.0 ** -np.array([[[0.1, 0.2, 0.3], [0.8, 1.0, 1.2], [0.8, 1.0, 6.0]]], np.float32)
    plain = invert_negative(rgb, BASE, d_hi=(0.8, 1.0, 1.2))
    res = invert_negative(rgb, BASE, d_hi=(0.8, 1.0, 1.2), spike_clamp=(1.15, 0.12))
    assert plain.positive[0, 2, 2] > 1e3 and res.positive[0, 2, 2] < 2.0
    np.testing.assert_array_equal(res.positive[0, :2], plain.positive[0, :2])
    assert res.params.to_dict()["spike_clamp"] == [1.15, 0.12] and "spike_clamp" not in plain.params.to_dict()
    assert res.report["spike_clamp"]["pixels"] == 1 and "spike_clamp" not in plain.report


def test_t4n_tiny_blue_white_point_stays_finite_without_changing_normal_pixels():
    p = InversionParams(base=tuple(BASE), d_hi=(0.632, 0.637, 0.098), spike_clamp=(1.15, 0.12))
    D = np.array([[[0.2, 0.3, 0.05], [3.0, 3.0, 6.0]]], np.float32)
    with np.errstate(over="raise", invalid="raise"):
        positive = invert(D, p)
    assert np.isfinite(positive).all() and positive.dtype == np.float32
    assert positive[0, 1, 2] > 1e10
    from filmcolor.holder import holder_mask
    assert holder_mask(positive)[0, 1]
    s = D[0, :1] / np.asarray(p.d_hi, np.float32)
    R = np.float32(p.log_range)
    floor = np.float32(10.0 ** -p.log_range)
    old = (np.power(np.float32(10.0), R * (s - 1)) - floor) / (1 - floor)
    np.testing.assert_array_equal(positive[0, :1], old)


def test_t4n_blue_guard_rejects_extreme_blue_collapse_but_keeps_warm_highlights():
    from filmcolor.invert import blue_white_guard

    guard = (1.06, 1.18, 0.65, 0.48)
    base = (0.5, 0.3, 0.1)
    for ratio in (0.154, 0.54999):
        d = (0.632, 1.0, ratio)
        out, info = blue_white_guard(d, guard, base=base)
        assert out == d and not info["applied"] and info["reason"] == "below_valid_ratio"
    for ratio in (0.55, 0.908, 1.038):
        out, info = blue_white_guard((0.632, 1.0, ratio), guard, base=base)
        assert info["applied"] and out[2] > ratio


def test_t4u_blue_white_guard_rescues_warm_highlight_narrow_blue_span_only():
    from filmcolor.invert import blue_white_guard

    guard = (1.06, 1.18, 0.65, 0.48)
    base_warm = (0.17, 0.225, 0.095)  # base_bg = 0.095 / 0.225 = 0.422 < 0.48

    # 1. Warm highlight with narrow blue span (A1006 signature):
    # d_hi = (0.92, 0.78, 0.086): ratio_bg = 0.086 / 0.78 = 0.110 < 0.28
    # Red-to-green ratio: d[0] / d[1] = 0.92 / 0.78 = 1.179 > 0.96
    # T4x bounded rescue:
    # d_b = 0.086 + 0.85 * (0.28 * 0.78 - 0.086) = 0.086 + 0.85 * (0.2184 - 0.086) = 0.19854
    d_warm = (0.92, 0.78, 0.086)
    out_warm, info_warm = blue_white_guard(d_warm, guard, base=base_warm)
    assert info_warm.get("applied") is True
    assert info_warm.get("narrow_blue_rescue") is True
    assert out_warm[0] == pytest.approx(0.92)
    assert out_warm[1] == pytest.approx(0.78)
    assert out_warm[2] == pytest.approx(0.19854, abs=1e-3)
    assert info_warm["d_hi_b_own"] == pytest.approx(0.086)
    assert info_warm["ratio_bg_used"] == pytest.approx(out_warm[2] / 0.78)

    # 2. Blue flare / extreme collapse with low R/G (d[0] <= 0.96 * d[1]):
    # d_flare = (0.632, 1.0, 0.154): d[0] / d[1] = 0.632 <= 0.96
    d_flare = (0.632, 1.0, 0.154)
    out_flare, info_flare = blue_white_guard(d_flare, guard, base=base_warm)
    assert out_flare == d_flare
    assert not info_flare.get("applied")
    assert info_flare.get("reason") == "below_valid_ratio"
    assert "narrow_blue_rescue" not in info_flare

    # 3. High blue/green base ratio (base_bg >= 0.48):
    base_cool = (0.50, 0.30, 0.20)  # base_bg = 0.20 / 0.30 = 0.667 >= 0.48
    out_cool_base, info_cool_base = blue_white_guard(d_warm, guard, base=base_cool)
    assert out_cool_base == d_warm
    assert not info_cool_base.get("applied")
    assert "narrow_blue_rescue" not in info_cool_base


def test_t4w_blue_white_guard_rescues_near_equal_rg_narrow_blue_but_rejects_flare():
    """T4w/T4x (fixes A1005.CR2 & A1007.CR2):
    1) d_hi = (0.632, 0.637, 0.098) (d[0]/d[1] = 0.992 > 0.96, ratio = 0.1538 < 0.28, A1005 signature)
       under base_bg < 0.48 triggers narrow_blue_rescue = True and lifts d_hi_B to bounded 0.166306.
    2) d_hi = (0.680, 0.696, 0.225) (ratio = 0.3233 >= 0.28, A1007 signature) under T4x is NOT rescued
       (applied is False, reason='below_valid_ratio', out_1007 == d_a1007) avoiding yellow-green overcorrection.
    3) Low R/G flare (0.632, 1.0, 0.154) (d[0]/d[1] = 0.632 <= 0.96) is still rejected (reason='below_valid_ratio')."""
    from filmcolor.invert import blue_white_guard

    guard = (1.06, 1.18, 0.65, 0.48)
    base_warm = (0.17, 0.225, 0.095)  # base_bg = 0.095 / 0.225 = 0.422 < 0.48

    # 1. A1005 signature: d_hi = (0.632, 0.637, 0.098), d[0]/d[1] = 0.9915 > 0.96, ratio = 0.1538 < 0.28
    d_a1005 = (0.632, 0.637, 0.098)
    out_1005, info_1005 = blue_white_guard(d_a1005, guard, base=base_warm)
    assert info_1005.get("applied") is True
    assert info_1005.get("narrow_blue_rescue") is True
    assert out_1005[0] == pytest.approx(0.632)
    assert out_1005[1] == pytest.approx(0.637)
    # T4x: d_b = 0.098 + 0.85 * (0.28 * 0.637 - 0.098) = 0.166306
    assert out_1005[2] == pytest.approx(0.166306, abs=1e-3)
    assert info_1005["d_hi_b_own"] == pytest.approx(0.098)
    assert info_1005["ratio_bg_used"] == pytest.approx(out_1005[2] / 0.637)

    # 2. A1007 signature: d_hi = (0.680, 0.696, 0.225), ratio = 0.225 / 0.696 = 0.3233 >= 0.28
    # In T4x, ratio >= 0.28 does not trigger narrow_blue_rescue, leaving clear sky/sea untouched.
    d_a1007 = (0.680, 0.696, 0.225)
    out_1007, info_1007 = blue_white_guard(d_a1007, guard, base=base_warm)
    assert out_1007 == d_a1007
    assert info_1007.get("applied") is False
    assert info_1007.get("reason") == "below_valid_ratio"
    assert "narrow_blue_rescue" not in info_1007

    # 3. Flare rejection: d_hi = (0.632, 1.0, 0.154), d[0]/d[1] = 0.632 <= 0.96
    d_flare = (0.632, 1.0, 0.154)
    out_flare, info_flare = blue_white_guard(d_flare, guard, base=base_warm)
    assert out_flare == d_flare
    assert not info_flare.get("applied")
    assert info_flare.get("reason") == "below_valid_ratio"
    assert "narrow_blue_rescue" not in info_flare


def test_t4x_blue_white_guard_caps_narrow_rescue_at_028_and_leaves_moderate_blue_untouched():
    """T4x: narrow blue rescue caps target at 0.28 * d_G and only triggers when ratio < 0.28.
    - Extreme collapsed ratios (0.11, 0.15) are rescued towards 0.28 * d_G (closing 85% of gap),
      reaching ~0.255..0.261 < 0.28.
    - Moderate blue spans (0.29, 0.32 >= 0.28) are untouched, avoiding yellow-green overcorrection."""
    from filmcolor.invert import blue_white_guard

    guard = (1.06, 1.18, 0.65, 0.48)
    base_warm = (0.17, 0.225, 0.095)  # base_bg = 0.422 < 0.48

    # 1. Extreme ratio = 0.1103 (e.g. A1006 signature: d_G = 0.78, d_B = 0.086):
    # Target cap: 0.28 * 0.78 = 0.2184
    # Rescued d_B = 0.086 + 0.85 * (0.2184 - 0.086) = 0.19854
    # ratio_bg_used = 0.19854 / 0.78 = 0.25454
    d1 = (0.92, 0.78, 0.086)
    out1, info1 = blue_white_guard(d1, guard, base=base_warm)
    assert info1.get("applied") is True
    assert info1.get("narrow_blue_rescue") is True
    assert out1[2] == pytest.approx(0.19854, abs=1e-4)
    assert 0.25 <= info1["ratio_bg_used"] <= 0.26

    # 2. Extreme ratio = 0.1538 (e.g. A1005 signature: d_G = 0.637, d_B = 0.098):
    # Target cap: 0.28 * 0.637 = 0.17836
    # Rescued d_B = 0.098 + 0.85 * (0.17836 - 0.098) = 0.166306
    # ratio_bg_used = 0.166306 / 0.637 = 0.261077
    d2 = (0.632, 0.637, 0.098)
    out2, info2 = blue_white_guard(d2, guard, base=base_warm)
    assert info2.get("applied") is True
    assert info2.get("narrow_blue_rescue") is True
    assert out2[2] == pytest.approx(0.166306, abs=1e-4)
    assert 0.26 <= info2["ratio_bg_used"] <= 0.265

    # 3. Moderate blue span with ratio = 0.29 >= 0.28 (d_G = 1.0, d_B = 0.29, d_R = 1.0):
    d3 = (1.0, 1.0, 0.29)
    out3, info3 = blue_white_guard(d3, guard, base=base_warm)
    assert out3 == d3
    assert info3.get("applied") is False
    assert info3.get("reason") == "below_valid_ratio"
    assert "narrow_blue_rescue" not in info3

    # 4. Moderate blue span with ratio = 0.3233 >= 0.28 (e.g. A1007 signature: d_G = 0.696, d_B = 0.225):
    d4 = (0.680, 0.696, 0.225)
    out4, info4 = blue_white_guard(d4, guard, base=base_warm)
    assert out4 == d4
    assert info4.get("applied") is False
    assert info4.get("reason") == "below_valid_ratio"
    assert "narrow_blue_rescue" not in info4


def test_t4l_self_ratio_white_point_lifts_only_a_channel_collapsed_against_its_own_midtones():
    """T4l: a white point channel ratio below collapse x the frame's own midtone ratio (DSC_3827: blue 0.91 against
    midtones about 1.3) is set to target x the midtone ratio; a scanner-like frame whose midtones and highlights share
    a low B / G (BlueNeg about 0.35) is unchanged (the T4k fixed 1.12 guard turned those yellow)."""
    from filmcolor.invert import self_ratio_white_point

    rng = np.random.default_rng(9)
    g = rng.uniform(0.25, 0.70, 5000)
    camera = np.stack([0.75 * g, g, 1.30 * g], -1)
    d, info = self_ratio_white_point((0.75, 1.0, 0.91), camera, (0.78, 0.85))
    assert info["applied"] and info["channels"] == ["B"]
    assert d[0] == 0.75 and d[1] == 1.0 and d[2] == pytest.approx(0.85 * 1.30)
    scanner = np.stack([0.60 * g, g, 0.35 * g], -1)
    d_s, info_s = self_ratio_white_point((0.60, 1.0, 0.35), scanner, (0.78, 0.85))
    assert d_s == (0.60, 1.0, 0.35) and not info_s["applied"]
    few = self_ratio_white_point((0.75, 1.0, 0.91), camera[:100], (0.78, 0.85))[1]
    assert not few["applied"] and few["n_mid"] < 500


def test_t4o_block_white_point_excludes_edge_connected_opaque_holder():
    """T4o: when initial d_hi_G > 1.80, opaque holder connected to any of the 4 grid edges (middle > 1.85,
    max > 2.10) dilated by 2 blocks is excluded from ok blocks; recalculates pct white point and logs excluded blocks."""
    from filmcolor.invert import MIN_WHITE_BLOCKS, block_white_point

    block = 16
    h, w = 512, 512
    rgb = np.zeros((h, w, 3), np.float32)
    D_scene = np.array([0.8, 1.0, 1.2], np.float32)
    rgb[:] = BASE * (10.0 ** -D_scene)

    # Wide holder on top 8 rows of blocks (rows 0:128)
    D_holder = np.array([2.0, 1.95, 2.3], np.float32)
    rgb[:128, :] = BASE * (10.0 ** -D_holder)

    # Central mask (rows 64:448, cols 64:448): spans 24x24 = 576 blocks, including 4 rows of holder
    mask = np.zeros((h, w), bool)
    mask[64:448, 64:448] = True

    d_hi, info = block_white_point(rgb, BASE, mask, block=block, pct=99.8)
    assert info["holder_blocks_excluded"] == 144  # 6 rows (4 in holder + 2 dilated) * 24 cols
    assert info["n_blocks"] == 432  # 576 - 144 >= MIN_WHITE_BLOCKS
    assert d_hi[1] < 1.5
    np.testing.assert_allclose(d_hi, D_scene, atol=1e-5)


def test_t4o_block_white_point_keeps_isolated_internal_high_density():
    """T4o: high-density blocks in the frame interior not connected to the image boundary are not excluded."""
    from filmcolor.invert import block_white_point

    block = 16
    h, w = 512, 512
    rgb = np.zeros((h, w, 3), np.float32)
    D_scene = np.array([0.8, 1.0, 1.2], np.float32)
    rgb[:] = BASE * (10.0 ** -D_scene)

    # Isolated high density in center (rows 256:288, cols 256:288): 2x2 blocks
    D_high = np.array([2.0, 1.95, 2.3], np.float32)
    rgb[256:288, 256:288] = BASE * (10.0 ** -D_high)

    mask = np.zeros((h, w), bool)
    mask[64:448, 64:448] = True

    d_hi, info = block_white_point(rgb, BASE, mask, block=block, pct=99.8)
    assert "holder_blocks_excluded" not in info
    assert info["n_blocks"] == 576
    np.testing.assert_allclose(d_hi, D_high, atol=1e-5)


def test_t4o_block_white_point_gated_by_d_hi_g():
    """T4o: when initial d_hi_G <= 1.80, the grid search is bypassed even if edge holder blocks exist."""
    from filmcolor.invert import block_white_point

    block = 16
    h, w = 512, 512
    rgb = np.zeros((h, w, 3), np.float32)
    D_scene = np.array([0.8, 1.0, 1.2], np.float32)
    rgb[:] = BASE * (10.0 ** -D_scene)

    # Edge holder outside the mask: rows 0:32 (2 rows of blocks)
    D_holder = np.array([2.0, 1.95, 2.3], np.float32)
    rgb[:32, :] = BASE * (10.0 ** -D_holder)

    # Mask only covers central rows 64:448 where D_G is 1.0 <= 1.80
    mask = np.zeros((h, w), bool)
    mask[64:448, 64:448] = True

    d_hi, info = block_white_point(rgb, BASE, mask, block=block, pct=99.8)
    assert "holder_blocks_excluded" not in info
    assert d_hi[1] == 1.0
    assert info["n_blocks"] == 576


def test_t4o_block_white_point_remaining_blocks_below_min_keeps_initial():
    """T4o: if excluded holder blocks leave ok_clean < MIN_WHITE_BLOCKS, initial d_hi is retained unchanged."""
    from filmcolor.invert import block_white_point

    block = 16
    h, w = 512, 512
    rgb = np.zeros((h, w, 3), np.float32)

    # Edge-connected holder across almost all the image
    D_holder = np.array([2.0, 1.95, 2.3], np.float32)
    rgb[:] = BASE * (10.0 ** -D_holder)

    # Tiny scene island of 8x10 = 80 blocks (< MIN_WHITE_BLOCKS = 100)
    D_scene = np.array([0.8, 1.0, 1.2], np.float32)
    rgb[200:200 + 8 * 16, 200:200 + 10 * 16] = BASE * (10.0 ** -D_scene)

    # Mask covers 12x12 = 144 blocks
    mask = np.zeros((h, w), bool)
    mask[160:352, 160:352] = True

    d_hi, info = block_white_point(rgb, BASE, mask, block=block, pct=99.8)
    assert "holder_blocks_excluded" not in info
    assert info["n_blocks"] == 144
    assert d_hi[1] > 1.80  # initial calculation preserved


def test_t4r_clamp_spikes_escape_rate_gating_and_holder_protection():
    """T4r: clamp_spikes gates secondary escape clamping (bound_z = max(min, 1) + margin) on escape.mean > 0.01.
    Tests:
    1. escape <= 1%: escape pixels are NOT clamped by bound_z (remain unclamped or governed by original bound).
    2. escape > 1%: escape pixels (min <= 1.6) are clamped to min(bound_z, bound).
    3. holder min > 1.6: retains original bound (e.g. inf when mid > mid_max), not clamped by bound_z."""
    from filmcolor.invert import clamp_spikes

    clamp = (1.15, 0.12)
    mid_max, margin = clamp

    # Case 1: escape <= 1% boundary (exact 2 / 200 = 1.0% <= 1% and 1 / 200 = 0.5% < 1%)
    n_pix = 200
    s_exact = np.full((n_pix, 3), [0.8, 0.9, 0.85], dtype=np.float32)
    # Exactly 2 escape pixels: mid=1.3 > 1.15, min=0.8 <= 1.6, max=6.0 > 5.0 -> 2/200 = 0.0100 (<= 1%)
    s_exact[:2] = [0.8, 1.3, 6.0]
    out_exact, count_exact = clamp_spikes(s_exact, clamp)
    # Since escape.mean == 0.010 <= 0.01, escape logic does not trigger (strictly > 0.01 needed)
    # mid=1.3 > mid_max -> bound=inf -> no clamping occurs
    assert count_exact == 0
    np.testing.assert_allclose(out_exact[:2], [[0.8, 1.3, 6.0], [0.8, 1.3, 6.0]], atol=1e-5)

    # Case 2: escape > 1% (e.g. 3 escape pixels out of 200 = 1.5% > 1%)
    s_high = np.full((n_pix, 3), [0.8, 0.9, 0.85], dtype=np.float32)
    # Escape pixels with min <= 1.6: e.g. [0.8, 1.3, 6.0] -> s_min=0.8, bound_z = max(0.8, 1.0) + 0.12 = 1.12
    s_high[:3] = [0.8, 1.3, 6.0]
    out_high, count_high = clamp_spikes(s_high, clamp)
    assert count_high >= 3
    # For pixel 0, max channel (6.0) is clamped to bound_z = 1.12 (min(1.12, inf) = 1.12)
    np.testing.assert_allclose(out_high[0], [0.8, 1.12, 1.12], atol=1e-5)

    # Case 3: opaque holder with min > 1.6 (e.g. [1.8, 2.5, 9.0])
    # Add a holder pixel into s_high where escape > 1%
    s_high_with_holder = s_high.copy()
    s_high_with_holder[10] = [1.8, 2.5, 9.0]
    out_holder, _ = clamp_spikes(s_high_with_holder, clamp)
    # For holder: min=1.8 > 1.6, so bound_z is not applied, bound remains inf because mid=2.5 > mid_max
    np.testing.assert_allclose(out_holder[10], [1.8, 2.5, 9.0], atol=1e-5)


def test_t4r_invert_negative_chunk_rows_local_escape_scope():
    """T4r: contract test documenting that invert_negative processes the image in CHUNK_ROWS (256-row blocks).
    If escape pixels are concentrated in one 256-row chunk (> 1% locally), that chunk triggers clamp_spikes,
    even if across the entire image escape pixels are <= 1%."""
    from filmcolor.invert import CHUNK_ROWS

    # Create a 2-chunk image: 512 rows x 100 cols = 51200 pixels
    h, w = CHUNK_ROWS * 2, 100
    D = np.full((h, w, 3), [0.8, 1.0, 1.0], dtype=np.float32)

    # Chunk 0 (rows 0:256): 25600 pixels. Put 300 escape pixels (300 / 25600 = 1.17% > 1% locally)
    # Escape in s: min <= 1.6, mid > 1.15, max > 5.0
    # For d_hi = (1.0, 1.0, 1.0) and d_lo = 0, s = D:
    D[:3, :100] = [0.8, 1.3, 6.0]  # 300 pixels with min=0.8, mid=1.3, max=6.0

    # Total escape pixels across full image = 300 / 51200 = 0.586% <= 1% globally!
    # Chunk 1 (rows 256:512): 0 escape pixels (0% locally).
    rgb = BASE * (10.0 ** -D)
    res = invert_negative(rgb, BASE, d_hi=(1.0, 1.0, 1.0), spike_clamp=(1.15, 0.12))

    # Because chunk 0 has local escape > 1%, clamp_spikes fires in chunk 0:
    # rows 0:3 have max channel (6.0) clamped to bound_z = 1.12, so positive is bounded.
    assert res.report["spike_clamp"]["pixels"] == 300
    assert res.positive[:3, :100, 2].max() < 2.0


def test_t4s_invert_negative_global_dm_zero_escape_consistency():
    """T4s: invert_negative checks zero_escape on global Dm before chunk loop.
    In a 512x100 image (2 chunks of 256 rows):
    - Chunk 0 has 4.0% escape pixels (1024 / 25600)
    - Chunk 1 has 0.5% escape pixels (128 / 25600)
    Globally Dm has (1024 + 128) / 51200 = 2.25% > 1.0% escape pixels.
    Global zero_escape propagates to Chunk 1 so its 0.5% escape pixels are also clamped
    by bound_z and do not blow up (> 1000)."""
    from filmcolor.invert import CHUNK_ROWS

    h, w = CHUNK_ROWS * 2, 100
    D = np.full((h, w, 3), [0.8, 1.0, 1.0], dtype=np.float32)

    # Chunk 0: rows 0:256, 25600 px -> 4% = 1024 px
    D_c0 = D[:CHUNK_ROWS].reshape(-1, 3)
    D_c0[:1024] = [0.8, 1.3, 6.0]

    # Chunk 1: rows 256:512, 25600 px -> 0.5% = 128 px
    D_c1 = D[CHUNK_ROWS:].reshape(-1, 3)
    D_c1[:128] = [0.8, 1.3, 6.0]

    rgb = BASE * (10.0 ** -D)
    res = invert_negative(rgb, BASE, d_hi=(1.0, 1.0, 1.0), spike_clamp=(1.15, 0.12))

    # Chunk 1 must be clamped via global zero_escape
    assert res.report["spike_clamp"]["pixels"] == 1024 + 128
    assert res.positive[CHUNK_ROWS:, :, 2].max() < 2.0


def test_t4s_invert_negative_chunk_rows_invariance_under_global_zero_escape(monkeypatch):
    """T4s: positive output is identical regardless of CHUNK_ROWS when global zero_escape is active."""
    import filmcolor.invert as inv

    h, w = 512, 100
    D = np.full((h, w, 3), [0.8, 1.0, 1.0], dtype=np.float32)
    # Global escape rate = (1024 + 128) / 51200 = 2.25% > 1.0%
    D[:256].reshape(-1, 3)[:1024] = [0.8, 1.3, 6.0]
    D[256:].reshape(-1, 3)[:128] = [0.8, 1.3, 6.0]
    rgb = BASE * (10.0 ** -D)

    monkeypatch.setattr(inv, "CHUNK_ROWS", 256)
    res_256 = inv.invert_negative(rgb, BASE, d_hi=(1.0, 1.0, 1.0), spike_clamp=(1.15, 0.12))

    monkeypatch.setattr(inv, "CHUNK_ROWS", 128)
    res_128 = inv.invert_negative(rgb, BASE, d_hi=(1.0, 1.0, 1.0), spike_clamp=(1.15, 0.12))

    np.testing.assert_allclose(res_256.positive, res_128.positive, atol=1e-6)
    assert res_256.report["spike_clamp"]["pixels"] == res_128.report["spike_clamp"]["pixels"] == 1152


def test_t4s_invert_negative_mask_without_escape_preserves_local_chunk_escape_scope():
    """T4s: when mask selects only clean pixels so global zero_escape is False, local chunks
    with > 1% escape pixels still trigger clamp_spikes via local fallback."""
    from filmcolor.invert import CHUNK_ROWS

    h, w = CHUNK_ROWS * 2, 100
    D = np.full((h, w, 3), [0.8, 1.0, 1.0], dtype=np.float32)
    # Chunk 0 has 300 escape pixels (300 / 25600 = 1.17% > 1% locally)
    D[:3, :100] = [0.8, 1.3, 6.0]
    rgb = BASE * (10.0 ** -D)

    # Mask only selects Chunk 1 (clean pixels, 0% escape, Dm has 0 escape -> zero_escape = False)
    mask = np.zeros((h, w), dtype=bool)
    mask[CHUNK_ROWS:] = True

    res = invert_negative(rgb, BASE, mask=mask, d_hi=(1.0, 1.0, 1.0), spike_clamp=(1.15, 0.12))

    # Chunk 0 still triggers local clamp_spikes because zero_escape=False retains local gating
    assert res.report["spike_clamp"]["pixels"] == 300
    assert res.positive[:3, :100, 2].max() < 2.0


def test_t4t_block_white_point_caps_extreme_single_channel_zero_clip():
    """T4t: block_white_point caps non-green channels whose density exceeds max(2.5, 2.2 * d_hi_G)
    to 1.45 * d_hi_G when d_hi_G >= 0.20, logging raw values in info['channel_cap'].
    Normal range white points and boundary conditions remain unchanged."""
    from filmcolor.invert import block_white_point

    block = 16
    h, w = 256, 256
    # 16x16 = 256 blocks total
    rgb = np.zeros((h, w, 3), np.float32)
    D_scene = np.array([0.40, 0.45, 0.50], np.float32)
    rgb[:] = BASE * (10.0 ** -D_scene)

    # In 2 blocks (~0.8% of blocks), simulate blue-channel zero-clip:
    # linear signal = 0 in blue -> D_B = 6.0
    D_clip = np.array([0.40, 0.45, 6.0], np.float32)
    rgb[0:block, 0:block * 2] = BASE * (10.0 ** -D_clip)

    mask = np.ones((h, w), bool)
    d_hi, info = block_white_point(rgb, BASE, mask, block=block, pct=99.8)

    # Blue channel was > max(2.5, 2.2 * 0.45 = 2.5) -> capped to 1.45 * 0.45 = 0.6525
    assert "channel_cap" in info
    assert "B" in info["channel_cap"]
    assert info["channel_cap"]["B"] > 2.5
    assert d_hi[2] == pytest.approx(1.45 * d_hi[1], abs=1e-5)
    assert d_hi[0] == pytest.approx(0.40, abs=1e-2)
    assert d_hi[1] == pytest.approx(0.45, abs=1e-2)

    # Normal white point unchanged
    rgb_normal = np.zeros((h, w, 3), np.float32)
    D_normal = np.array([0.80, 1.00, 1.35], np.float32)
    rgb_normal[:] = BASE * (10.0 ** -D_normal)
    d_hi_norm, info_norm = block_white_point(rgb_normal, BASE, mask, block=block, pct=99.8)
    assert "channel_cap" not in info_norm
    np.testing.assert_allclose(d_hi_norm, D_normal, atol=1e-4)

    # Boundary: d_hi_G < 0.20 must NOT trigger channel_cap even if R/B is high
    rgb_low_g = np.zeros((h, w, 3), np.float32)
    D_low_g = np.array([0.15, 0.18, 3.0], np.float32)
    rgb_low_g[:] = BASE * (10.0 ** -D_low_g)
    d_hi_low, info_low = block_white_point(rgb_low_g, BASE, mask, block=block, pct=99.8)
    assert "channel_cap" not in info_low


def test_t4t_block_white_point_excludes_surrounding_slide_mount():
    """T4t: when initial d_hi_G > 1.15 and d_hi_R > d_hi_G + 0.25 (slide mount signature),
    and edge ring of cand blocks (D_sort[1] > 1.05 & D_sort[2] > 1.45) has mean >= 0.75,
    connected border is excluded; internal islands and partial rings (< 0.75) are preserved,
    and MIN_WHITE_BLOCKS constraint is respected."""
    from filmcolor.invert import MIN_WHITE_BLOCKS, block_white_point

    block = 16
    h, w = 512, 512
    # 32x32 = 1024 blocks total

    # 1. Full surrounding slide mount:
    # Border 4 blocks thick on all 4 sides (rows 0:64, 448:512, cols 0:64, 448:512)
    rgb = np.zeros((h, w, 3), np.float32)
    D_scene = np.array([0.85, 0.90, 1.00], np.float32)
    rgb[:] = BASE * (10.0 ** -D_scene)

    # Mount: D = [1.70, 1.25, 1.50], so D_sort = [1.25, 1.50, 1.70] -> D_sort[1]=1.50 > 1.05 & D_sort[2]=1.70 > 1.45
    # d_hi_G = 1.25 > 1.15, d_hi_R = 1.70 > 1.25 + 0.25 = 1.50
    D_mount = np.array([1.70, 1.25, 1.50], np.float32)
    rgb[:64, :] = BASE * (10.0 ** -D_mount)
    rgb[448:, :] = BASE * (10.0 ** -D_mount)
    rgb[:, :64] = BASE * (10.0 ** -D_mount)
    rgb[:, 448:] = BASE * (10.0 ** -D_mount)

    mask = np.ones((h, w), bool)
    d_hi, info = block_white_point(rgb, BASE, mask, block=block, pct=99.8)

    assert info.get("holder_blocks_excluded", 0) > 0
    np.testing.assert_allclose(d_hi, D_scene, atol=0.05)

    # 2. Incomplete ring (< 0.75 on edge ring): e.g. only top edge has mount blocks
    rgb_partial = np.zeros((h, w, 3), np.float32)
    rgb_partial[:] = BASE * (10.0 ** -D_scene)
    rgb_partial[:64, :] = BASE * (10.0 ** -D_mount)
    # Edge ring mean will be ~ 32 / (32*4 - 4) ~ 0.26 < 0.75 -> must NOT peel
    d_hi_part, info_part = block_white_point(rgb_partial, BASE, mask, block=block, pct=99.8)
    assert "holder_blocks_excluded" not in info_part

    # 3. Isolated internal high density island not touching edge: must NOT peel
    rgb_island = np.zeros((h, w, 3), np.float32)
    rgb_island[:] = BASE * (10.0 ** -D_scene)
    rgb_island[200:300, 200:300] = BASE * (10.0 ** -D_mount)
    d_hi_isl, info_isl = block_white_point(rgb_island, BASE, mask, block=block, pct=99.8)
    assert "holder_blocks_excluded" not in info_isl


def test_t4t_invert_negative_dm_zero_escape_at_sub_percent_rate():
    """T4t: invert_negative checks zero_escape on global Dm with threshold > 0.001 (0.1%),
    so an escape rate of 0.35% (e.g. A305.tif, > 0.001 but <= 0.01) triggers global zero_escape = True,
    clamping positive escape pixels to < 2.0; while strict rate <= 0.1% does not trigger."""
    from filmcolor.invert import CHUNK_ROWS

    h, w = CHUNK_ROWS * 2, 100
    n_pix = h * w  # 51200
    D = np.full((h, w, 3), [0.8, 1.0, 1.0], dtype=np.float32)

    # 0.35% of 51200 = 179 pixels distributed across chunk 0 and chunk 1
    # Chunk 0 has 100 escape pixels (100 / 25600 = 0.39% <= 1% locally)
    # Chunk 1 has 79 escape pixels (79 / 25600 = 0.31% <= 1% locally)
    # Neither chunk has > 1% locally! But globally 179 / 51200 = 0.3496% > 0.1% (0.001).
    D[:CHUNK_ROWS].reshape(-1, 3)[:100] = [0.8, 1.3, 6.0]
    D[CHUNK_ROWS:].reshape(-1, 3)[:79] = [0.8, 1.3, 6.0]

    rgb = BASE * (10.0 ** -D)
    res = invert_negative(rgb, BASE, d_hi=(1.0, 1.0, 1.0), spike_clamp=(1.15, 0.12))

    # Under T4t (threshold 0.001), zero_escape triggers globally, clamping all 179 pixels:
    assert res.report["spike_clamp"]["pixels"] == 179
    assert res.positive[:CHUNK_ROWS, :, 2].max() < 2.0
    assert res.positive[CHUNK_ROWS:, :, 2].max() < 2.0

    # Strict sub-threshold boundary: 0.08% (40 / 51200 = 0.00078 <= 0.001) must NOT trigger global zero_escape
    D_sub = np.full((h, w, 3), [0.8, 1.0, 1.0], dtype=np.float32)
    D_sub[:CHUNK_ROWS].reshape(-1, 3)[:20] = [0.8, 1.3, 6.0]
    D_sub[CHUNK_ROWS:].reshape(-1, 3)[:20] = [0.8, 1.3, 6.0]
    rgb_sub = BASE * (10.0 ** -D_sub)
    res_sub = invert_negative(rgb_sub, BASE, d_hi=(1.0, 1.0, 1.0), spike_clamp=(1.15, 0.12))
    # Neither chunk has > 1% locally, and globally <= 0.1% -> no clamping
    assert res_sub.report["spike_clamp"]["pixels"] == 0
    assert res_sub.positive[..., 2].max() > 10.0


def test_t4z_white_point_bridges_one_block_mullion():
    """T4z: block_white_point bridges a 1-block window mullion when edge holder blocks exist,
    excluding internal hot blocks separated by thin mullions; baseline with internal hot island alone remains unchanged."""
    from filmcolor.invert import block_white_point
    block = 16
    h, w = 512, 512
    mask = np.ones((h, w), bool)

    D_scene = np.array([0.85, 0.90, 1.00], np.float32)
    D_hot = np.array([2.2, 2.2, 2.2], np.float32)

    # 1. Edge-connected hot blocks with internal hot block separated by 1-block window mullion
    # Top hot touches only top edge (rows 0:32, cols 128:384); gap 32:48; internal hot (rows 48:144, cols 128:384)
    # Internal hot is well beyond dilation(iterations=2) from rows 0:32 and detached from left/right edges.
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[:] = BASE * (10.0 ** -D_scene)
    rgb[:32, 128:384] = BASE * (10.0 ** -D_hot)
    rgb[48:144, 128:384] = BASE * (10.0 ** -D_hot)

    d_hi, info = block_white_point(rgb, BASE, mask, block=block, pct=99.8)
    assert info.get("holder_blocks_excluded", 0) > 0
    np.testing.assert_allclose(d_hi, D_scene, atol=0.01)

    # 2. Baseline without edge hot blocks: isolated internal hot island alone does not trigger holder peeling
    rgb_island = np.zeros((h, w, 3), np.float32)
    rgb_island[:] = BASE * (10.0 ** -D_scene)
    rgb_island[100:200, 100:200] = BASE * (10.0 ** -D_hot)
    d_hi_isl, info_isl = block_white_point(rgb_island, BASE, mask, block=block, pct=99.8)
    assert "holder_blocks_excluded" not in info_isl
    np.testing.assert_allclose(d_hi_isl, [2.2, 2.2, 2.2], atol=0.01)


def test_t5e_block_white_point_caps_inflated_blue_ratio():
    """T5e (A831/A832): when d_hi_G >= 0.55 and d_hi_B > 1.95 * d_hi_G (even if <= 2.2 * d_hi_G and <= 2.5 D),
    block_white_point caps d_hi_B to 1.45 * d_hi_G and logs info['channel_cap']['B']; when d_hi_B / d_hi_G = 1.85
    or d_hi_G < 0.55 it does not trigger."""
    from filmcolor.invert import block_white_point

    block = 16
    h, w = 256, 256
    mask = np.ones((h, w), bool)

    # 1. Trigger: d_hi_G = 0.80 (>= 0.55), d_hi_B = 1.62 (~2.025 * 0.80 > 1.95 * 0.80, but < 2.2 * 0.80 = 1.76 and < 2.5)
    rgb = np.zeros((h, w, 3), np.float32)
    D_scene = np.array([0.35, 0.80, 1.05], np.float32)
    rgb[:] = BASE * (10.0 ** -D_scene)
    D_sky = np.array([0.35, 0.80, 1.62], np.float32)
    rgb[0:block, 0:block * 2] = BASE * (10.0 ** -D_sky)

    d_hi, info = block_white_point(rgb, BASE, mask, block=block, pct=99.8)
    assert "channel_cap" in info and "B" in info["channel_cap"]
    assert info["channel_cap"]["B"] == pytest.approx(1.62, abs=1e-3)
    assert d_hi[1] == pytest.approx(0.80, abs=1e-4)
    assert d_hi[2] == pytest.approx(1.45 * d_hi[1], abs=1e-4)

    # 2. Non-trigger: d_hi_B / d_hi_G = 1.85 (<= 1.95)
    rgb_ok = np.zeros((h, w, 3), np.float32)
    rgb_ok[:] = BASE * (10.0 ** -np.array([0.35, 0.80, 1.48], np.float32))
    d_hi_ok, info_ok = block_white_point(rgb_ok, BASE, mask, block=block, pct=99.8)
    assert "channel_cap" not in info_ok
    assert d_hi_ok[2] == pytest.approx(1.48, abs=1e-4)

    # 3. Non-trigger: d_hi_G = 0.50 (< 0.55) with d_hi_B = 1.00 (2.0 * d_hi_G, < 2.5)
    rgb_low_g = np.zeros((h, w, 3), np.float32)
    rgb_low_g[:] = BASE * (10.0 ** -np.array([0.25, 0.50, 1.00], np.float32))
    d_hi_lg, info_lg = block_white_point(rgb_low_g, BASE, mask, block=block, pct=99.8)
    assert "channel_cap" not in info_lg


def test_t5f_fine_highlight_rescue_recovers_subblock_canopy_gaps():
    """T5f (A623.tif): when small bright canopy sky gaps (< 16x16 px) are diluted in 16x16 block means
    so d4_990_G - d_hi_G >= 0.10 and d4_990_G / d_hi_G >= 1.15 (with d_hi_G >= 0.45 and no holder_blocks_excluded),
    block_white_point updates d_hi to the 99.5th percentile of 4x4 sub-blocks and logs fine_highlight_rescue.
    Contiguous 16x16 highlights or thin negatives (d_hi_G < 0.45) do not trigger."""
    from filmcolor.invert import block_white_point

    block = 16
    h, w = 256, 256
    mask = np.ones((h, w), bool)

    # 1. Sub-block canopy gaps in ~20% of 16x16 blocks (52 / 256 blocks -> 52 / 4096 = 1.27% of 4x4 sub-blocks)
    rgb = np.zeros((h, w, 3), np.float32)
    D_bg = np.array([0.25, 0.55, 0.70], np.float32)
    D_gap = np.array([0.55, 1.15, 1.45], np.float32)
    rgb[:] = BASE * (10.0 ** -D_bg)
    count = 0
    for br in range(h // block):
        for bc in range(w // block):
            if count < 52:
                r0, c0 = br * block, bc * block
                rgb[r0 : r0 + 4, c0 : c0 + 4] = BASE * (10.0 ** -D_gap)
                count += 1

    d_hi, info = block_white_point(rgb, BASE, mask, block=block, pct=99.8)
    assert "fine_highlight_rescue" in info
    fhr = info["fine_highlight_rescue"]
    assert fhr["sub_block"] == 4 and fhr["pct"] == 99.5
    assert fhr["d_hi_own"][1] == pytest.approx(0.5708, abs=0.02)
    np.testing.assert_allclose(d_hi, D_gap, atol=1e-4)

    # 2. Contiguous 16x16 uniform highlight blocks: 16x16 white point already captures D_gap -> does not trigger
    rgb_contig = np.zeros((h, w, 3), np.float32)
    rgb_contig[:] = BASE * (10.0 ** -D_bg)
    rgb_contig[:block, : block * 2] = BASE * (10.0 ** -D_gap)
    d_hi_c, info_c = block_white_point(rgb_contig, BASE, mask, block=block, pct=99.8)
    assert "fine_highlight_rescue" not in info_c
    np.testing.assert_allclose(d_hi_c, D_gap, atol=1e-4)

    # 3. Thin negative (d_hi_G < 0.45): does not trigger
    rgb_thin = np.zeros((h, w, 3), np.float32)
    rgb_thin[:] = BASE * (10.0 ** -np.array([0.15, 0.30, 0.35], np.float32))
    count = 0
    for br in range(h // block):
        for bc in range(w // block):
            if count < 52:
                r0, c0 = br * block, bc * block
                rgb_thin[r0 : r0 + 4, c0 : c0 + 4] = BASE * (10.0 ** -np.array([0.35, 0.70, 0.80], np.float32))
                count += 1
    d_hi_t, info_t = block_white_point(rgb_thin, BASE, mask, block=block, pct=99.8)
    assert d_hi_t[1] < 0.45
    assert "fine_highlight_rescue" not in info_t


def test_t5f_zero_escape_clamps_moderate_8bit_floor_spikes():
    """T5f (A621/A623): when holder_blocks_excluded is absent, invert_negative's zero_escape precheck uses
    max_thr = 2.0 (instead of 5.0), clamping 0.3% pixels with s = [1.25, 1.30, 2.50] to s_B = 1.25 + 0.12 = 1.37;
    when holder_blocks_excluded is present, max_thr remains 5.0 so s_max = 2.50 does not trigger zero_escape."""
    h, w = 256, 256
    mask = np.ones((h, w), bool)
    D = np.full((h, w, 3), 0.50, dtype=np.float32)
    # 4 uniform 16x16 highlight blocks establish d_hi = [1.0, 1.0, 1.0]
    D[0:16, 0:64] = [1.0, 1.0, 1.0]
    # 197 scattered single pixels (~0.30% > 0.1%, <= 1.0% per chunk) at s = [1.25, 1.30, 2.50]
    placed = 0
    for br in range(1, h // 16):
        for bc in range(w // 16):
            if placed < 197:
                D[br * 16 + (placed % 4), bc * 16] = [1.25, 1.30, 2.50]
                placed += 1
    rgb = BASE * (10.0 ** -D)

    res = invert_negative(rgb, BASE, mask=mask, white_block=16, spike_clamp=(1.15, 0.12))
    assert "holder_blocks_excluded" not in res.report["white_point"]
    np.testing.assert_allclose(res.params.d_hi, [1.0, 1.0, 1.0], atol=1e-2)
    assert res.report["spike_clamp"]["pixels"] == 197
    # Clamped s_B = 1.25 + 0.12 = 1.37 -> matches unclamped invert at D_B = 1.37 * d_hi_B
    from filmcolor.invert import InversionParams, invert

    p_no_clamp = InversionParams(base=res.params.base, d_lo=res.params.d_lo, d_hi=res.params.d_hi,
                                 gamma=res.params.gamma, spike_clamp=None)
    expected_b = invert(np.array([[1.25, 1.30, 1.37 * res.params.d_hi[2]]], np.float32), p_no_clamp)[0, 2]
    assert res.positive[16, 0, 2] == pytest.approx(float(expected_b), rel=1e-3)

    # With holder_blocks_excluded present (512x512 image, top 2 block rows are hot holder [2.2, 2.2, 2.2]),
    # max_thr stays 5.0 so 0.3% interior pixels at s = [1.25, 1.30, 2.50] (< 5.0) do not trigger zero_escape
    h2, w2 = 512, 512
    mask2 = np.ones((h2, w2), bool)
    D2 = np.full((h2, w2, 3), 0.50, dtype=np.float32)
    D2[:32, :] = [2.2, 2.2, 2.2]
    D2[64:80, 0:128] = [1.0, 1.0, 1.0]  # 8 blocks -> > 0.2% of ~960 clean blocks so d_hi = [1.0, 1.0, 1.0]
    # Place 400 pixels in chunk 0 (rows 96:256) and 400 in chunk 1 (rows 256:512): 800 / 262144 = 0.305%, each chunk < 1%
    D2[96:100, :100] = [1.25, 1.30, 2.50]
    D2[256:260, :100] = [1.25, 1.30, 2.50]
    rgb2 = BASE * (10.0 ** -D2)

    res2 = invert_negative(rgb2, BASE, mask=mask2, white_block=16, spike_clamp=(1.15, 0.12))
    assert res2.report["white_point"].get("holder_blocks_excluded", 0) > 0
    np.testing.assert_allclose(res2.params.d_hi, [1.0, 1.0, 1.0], atol=1e-2)
    assert res2.report["spike_clamp"]["pixels"] == 0
    p2_no_clamp = InversionParams(base=res2.params.base, d_lo=res2.params.d_lo, d_hi=res2.params.d_hi,
                                  gamma=res2.params.gamma, spike_clamp=None)
    unclamped_b = invert(np.array([[1.25, 1.30, 2.50]], np.float32), p2_no_clamp)[0, 2]
    assert res2.positive[96, 0, 2] == pytest.approx(float(unclamped_b), rel=1e-3)
    assert res2.positive[96, 0, 2] > 200.0


