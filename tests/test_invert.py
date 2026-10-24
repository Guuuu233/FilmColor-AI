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
